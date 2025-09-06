package main

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.etcd.io/bbolt"
)

// Vector represents a point in a multi-dimensional space.
type Vector []float64

// Item represents a single data entry in our database, with a unique ID and its vector representation.
type Item struct {
	ID     string `json:"id"`
	Vector Vector `json:"vector"`
}

// SearchResult holds the result of a search query, containing the item and its distance to the query vector.
type SearchResult struct {
	Item     Item    `json:"item"`
	Distance float64 `json:"distance"`
}

// LSHBucket represents a bucket in the LSH index
type LSHBucket struct {
	Items []string // Item IDs in this bucket
}

// LSHIndex implements Locality Sensitive Hashing for hyperbolic space
type LSHIndex struct {
	NumHashFunctions int
	NumBuckets       int
	HashFunctions    []HashFunction
	Buckets          map[uint64]*LSHBucket
	mu               sync.RWMutex
}

// HashFunction represents a single LSH hash function
type HashFunction struct {
	RandomVector Vector
	Offset       float64
}

// HyperbolicDB is a vector database that uses hyperbolic distance with ANN indexing and persistence.
type HyperbolicDB struct {
	mu                    sync.RWMutex
	items                 map[string]Item
	lshIndex              *LSHIndex
	boltDB                *bbolt.DB
	dataPath              string
	indexRebuildThreshold int
	itemCount             int
}

// Config holds configuration options for the database
type Config struct {
	DataPath              string
	NumHashFunctions      int
	NumBuckets            int
	IndexRebuildThreshold int
}

// DefaultConfig returns a default configuration
func DefaultConfig() *Config {
	return &Config{
		DataPath:              "./hyperbolic_db",
		NumHashFunctions:      16,
		NumBuckets:            1000,
		IndexRebuildThreshold: 1000,
	}
}

// NewHyperbolicDB creates and returns a new instance of HyperbolicDB with persistence.
func NewHyperbolicDB(config *Config) (*HyperbolicDB, error) {
	if config == nil {
		config = DefaultConfig()
	}

	// Create data directory if it doesn't exist
	if err := os.MkdirAll(config.DataPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %v", err)
	}

	// Open BoltDB for persistence
	dbPath := filepath.Join(config.DataPath, "vectors.db")
	boltDB, err := bbolt.Open(dbPath, 0600, &bbolt.Options{Timeout: 1 * time.Second})
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %v", err)
	}

	// Initialize buckets
	err = boltDB.Update(func(tx *bbolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte("items"))
		if err != nil {
			return fmt.Errorf("create bucket: %s", err)
		}
		_, err = tx.CreateBucketIfNotExists([]byte("index"))
		return err
	})
	if err != nil {
		boltDB.Close()
		return nil, err
	}

	db := &HyperbolicDB{
		items:                 make(map[string]Item),
		boltDB:                boltDB,
		dataPath:              config.DataPath,
		indexRebuildThreshold: config.IndexRebuildThreshold,
	}

	// Initialize LSH index
	db.lshIndex = NewLSHIndex(config.NumHashFunctions, config.NumBuckets)

	// Load existing data from disk
	if err := db.loadFromDisk(); err != nil {
		boltDB.Close()
		return nil, fmt.Errorf("failed to load data: %v", err)
	}

	return db, nil
}

// NewLSHIndex creates a new LSH index for hyperbolic space
func NewLSHIndex(numHashFunctions, numBuckets int) *LSHIndex {
	rand.Seed(time.Now().UnixNano())

	index := &LSHIndex{
		NumHashFunctions: numHashFunctions,
		NumBuckets:       numBuckets,
		HashFunctions:    make([]HashFunction, numHashFunctions),
		Buckets:          make(map[uint64]*LSHBucket),
	}

	// Initialize hash functions with random vectors
	// For hyperbolic space, we use random unit vectors in the Poincaré ball
	for i := 0; i < numHashFunctions; i++ {
		// Generate random vector in higher dimensions for better separation
		dim := 64 // Use higher dimensional random vectors
		randomVec := make(Vector, dim)
		for j := 0; j < dim; j++ {
			randomVec[j] = rand.NormFloat64()
		}

		// Normalize to be inside Poincaré ball
		norm := vectorNorm(randomVec)
		for j := 0; j < dim; j++ {
			randomVec[j] = randomVec[j] / (norm + 1e-9) * 0.8 // Scale to 0.8 to stay well inside
		}

		index.HashFunctions[i] = HashFunction{
			RandomVector: randomVec,
			Offset:       rand.Float64() * 2 * math.Pi,
		}
	}

	return index
}

// hash computes LSH hash for a vector
func (lsh *LSHIndex) hash(v Vector) uint64 {
	var hash uint64 = 14695981039346656037 // FNV offset basis

	for i, hashFunc := range lsh.HashFunctions {
		// Project vector onto hash function space
		var dotProduct float64
		minDim := len(v)
		if len(hashFunc.RandomVector) < minDim {
			minDim = len(hashFunc.RandomVector)
		}

		for j := 0; j < minDim; j++ {
			dotProduct += v[j] * hashFunc.RandomVector[j]
		}

		// Use hyperbolic-aware hashing
		hyperHash := math.Floor((dotProduct + hashFunc.Offset) * float64(lsh.NumBuckets) / (2 * math.Pi))
		bucketHash := uint64(int(hyperHash) % lsh.NumBuckets)

		// Combine with existing hash using FNV-1a
		hash ^= bucketHash
		hash *= 1099511628211 // FNV prime
		hash ^= uint64(i)     // Include hash function index
		hash *= 1099511628211
	}

	return hash
}

// Add adds an item to the LSH index
func (lsh *LSHIndex) Add(item Item) {
	lsh.mu.Lock()
	defer lsh.mu.Unlock()

	hash := lsh.hash(item.Vector)

	if lsh.Buckets[hash] == nil {
		lsh.Buckets[hash] = &LSHBucket{Items: make([]string, 0)}
	}

	lsh.Buckets[hash].Items = append(lsh.Buckets[hash].Items, item.ID)
}

// GetCandidates returns candidate items for ANN search
func (lsh *LSHIndex) GetCandidates(queryVector Vector, maxCandidates int) []string {
	lsh.mu.RLock()
	defer lsh.mu.RUnlock()

	hash := lsh.hash(queryVector)
	candidateSet := make(map[string]bool)

	// Get items from the same bucket
	if bucket := lsh.Buckets[hash]; bucket != nil {
		for _, itemID := range bucket.Items {
			candidateSet[itemID] = true
		}
	}

	// If we don't have enough candidates, expand search to nearby buckets
	if len(candidateSet) < maxCandidates {
		// Simple expansion: try a few hash variations
		for i := 0; i < 10 && len(candidateSet) < maxCandidates; i++ {
			altHash := hash ^ uint64(i) // Simple bit flip for nearby buckets
			if bucket := lsh.Buckets[altHash]; bucket != nil {
				for _, itemID := range bucket.Items {
					candidateSet[itemID] = true
				}
			}
		}
	}

	// Convert set to slice
	candidates := make([]string, 0, len(candidateSet))
	for id := range candidateSet {
		candidates = append(candidates, id)
		if len(candidates) >= maxCandidates {
			break
		}
	}

	return candidates
}

// Rebuild rebuilds the entire LSH index
func (lsh *LSHIndex) Rebuild(items map[string]Item) {
	lsh.mu.Lock()
	defer lsh.mu.Unlock()

	// Clear existing buckets
	lsh.Buckets = make(map[uint64]*LSHBucket)

	// Re-add all items
	for _, item := range items {
		hash := lsh.hash(item.Vector)
		if lsh.Buckets[hash] == nil {
			lsh.Buckets[hash] = &LSHBucket{Items: make([]string, 0)}
		}
		lsh.Buckets[hash].Items = append(lsh.Buckets[hash].Items, item.ID)
	}
}

// Add inserts or updates an item in the database with persistence.
func (db *HyperbolicDB) Add(item Item) (string, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	if item.ID == "" {
		item.ID = uuid.New().String()
	}

	// Basic validation
	if len(item.Vector) == 0 {
		return "", fmt.Errorf("cannot add item with an empty vector")
	}

	// Normalize vector for Poincaré ball
	norm := vectorNorm(item.Vector)
	if norm >= 1.0 {
		epsilon := 1e-9
		for i := range item.Vector {
			item.Vector[i] /= (norm + epsilon)
		}
	}

	// Add to in-memory store
	_, wasNew := db.items[item.ID]
	wasNew = !wasNew // Invert because we want to know if it's new
	db.items[item.ID] = item

	// Add to LSH index
	db.lshIndex.Add(item)

	// Persist to disk
	err := db.boltDB.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte("items"))

		// Encode item
		var buf []byte
		enc := gob.NewEncoder(&gobBuffer{&buf})
		if err := enc.Encode(item); err != nil {
			return err
		}

		return b.Put([]byte(item.ID), buf)
	})

	if err != nil {
		// Rollback in-memory changes
		if wasNew {
			delete(db.items, item.ID)
		}
		return "", fmt.Errorf("failed to persist item: %v", err)
	}

	if wasNew {
		db.itemCount++
		// Rebuild index periodically for better performance
		if db.itemCount%db.indexRebuildThreshold == 0 {
			go db.rebuildIndex() // Async rebuild
		}
	}

	return item.ID, nil
}

// Search finds the k-nearest neighbors using ANN with LSH indexing.
func (db *HyperbolicDB) Search(queryVector Vector, k int) ([]SearchResult, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(db.items) == 0 {
		return []SearchResult{}, nil
	}

	// Normalize query vector
	norm := vectorNorm(queryVector)
	if norm >= 1.0 {
		epsilon := 1e-9
		for i := range queryVector {
			queryVector[i] /= (norm + epsilon)
		}
	}

	// Get candidates using LSH index (ANN approach)
	maxCandidates := k * 10 // Search 10x more candidates than needed
	if maxCandidates < 100 {
		maxCandidates = 100
	}
	if maxCandidates > len(db.items) {
		maxCandidates = len(db.items)
	}

	candidates := db.lshIndex.GetCandidates(queryVector, maxCandidates)

	// If we don't have enough candidates, fall back to brute force
	if len(candidates) < k*2 {
		candidates = make([]string, 0, len(db.items))
		for id := range db.items {
			candidates = append(candidates, id)
		}
	}

	var results []SearchResult

	// Calculate distances for candidates only
	for _, candidateID := range candidates {
		item, exists := db.items[candidateID]
		if !exists {
			continue
		}

		// Skip if dimensions don't match
		if len(item.Vector) != len(queryVector) {
			continue
		}

		dist := poincareDistance(item.Vector, queryVector)
		results = append(results, SearchResult{
			Item:     item,
			Distance: dist,
		})
	}

	// Sort by distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	// Return top k results
	if k > len(results) {
		k = len(results)
	}

	return results[:k], nil
}

// ExactSearch performs exact brute-force search (for comparison/verification)
func (db *HyperbolicDB) ExactSearch(queryVector Vector, k int) ([]SearchResult, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(db.items) == 0 {
		return []SearchResult{}, nil
	}

	// Normalize query vector
	norm := vectorNorm(queryVector)
	if norm >= 1.0 {
		epsilon := 1e-9
		for i := range queryVector {
			queryVector[i] /= (norm + epsilon)
		}
	}

	var results []SearchResult

	for _, item := range db.items {
		if len(item.Vector) != len(queryVector) {
			continue
		}

		dist := poincareDistance(item.Vector, queryVector)
		results = append(results, SearchResult{
			Item:     item,
			Distance: dist,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	if k > len(results) {
		k = len(results)
	}

	return results[:k], nil
}

// rebuildIndex rebuilds the LSH index (called periodically)
func (db *HyperbolicDB) rebuildIndex() {
	log.Println("Rebuilding LSH index...")
	start := time.Now()

	db.mu.RLock()
	items := make(map[string]Item, len(db.items))
	for k, v := range db.items {
		items[k] = v
	}
	db.mu.RUnlock()

	db.lshIndex.Rebuild(items)

	log.Printf("Index rebuild completed in %v", time.Since(start))
}

// loadFromDisk loads all persisted data from BoltDB
func (db *HyperbolicDB) loadFromDisk() error {
	return db.boltDB.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte("items"))
		if b == nil {
			return nil // No data to load
		}

		c := b.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			var item Item
			buf := v
			dec := gob.NewDecoder(&gobBuffer{&buf})
			if err := dec.Decode(&item); err != nil {
				log.Printf("Warning: failed to decode item %s: %v", string(k), err)
				continue
			}

			db.items[item.ID] = item
			db.lshIndex.Add(item)
			db.itemCount++
		}

		log.Printf("Loaded %d items from disk", len(db.items))
		return nil
	})
}

// Close closes the database and releases resources
func (db *HyperbolicDB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.boltDB != nil {
		return db.boltDB.Close()
	}
	return nil
}

// Stats returns database statistics
func (db *HyperbolicDB) Stats() map[string]interface{} {
	db.mu.RLock()
	defer db.mu.RUnlock()

	db.lshIndex.mu.RLock()
	numBuckets := len(db.lshIndex.Buckets)
	db.lshIndex.mu.RUnlock()

	return map[string]interface{}{
		"total_items":    len(db.items),
		"lsh_buckets":    numBuckets,
		"hash_functions": db.lshIndex.NumHashFunctions,
	}
}

// gobBuffer is a helper for gob encoding/decoding
type gobBuffer struct {
	data *[]byte
}

func (g *gobBuffer) Write(p []byte) (n int, err error) {
	*g.data = append(*g.data, p...)
	return len(p), nil
}

func (g *gobBuffer) Read(p []byte) (n int, err error) {
	if len(*g.data) == 0 {
		return 0, fmt.Errorf("EOF")
	}
	n = copy(p, *g.data)
	*g.data = (*g.data)[n:]
	return n, nil
}

// --- Utility and Math Functions ---

// vectorNorm calculates the L2 (Euclidean) norm of a vector.
func vectorNorm(v Vector) float64 {
	var sumOfSquares float64
	for _, val := range v {
		sumOfSquares += val * val
	}
	return math.Sqrt(sumOfSquares)
}

// poincareDistance calculates the geodesic distance between two vectors in the Poincaré ball model.
func poincareDistance(u, v Vector) float64 {
	sqNormU := vectorNorm(u) * vectorNorm(u)
	sqNormV := vectorNorm(v) * vectorNorm(v)

	diff := make(Vector, len(u))
	for i := range u {
		diff[i] = u[i] - v[i]
	}
	sqNormDiff := vectorNorm(diff) * vectorNorm(diff)

	denominator := (1 - sqNormU) * (1 - sqNormV)
	if denominator <= 0 {
		return math.MaxFloat64
	}

	argument := 1 + 2*(sqNormDiff/denominator)
	if argument < 1.0 {
		argument = 1.0
	}

	return math.Acosh(argument)
}

// --- HTTP API Handlers ---

// AddRequest defines the JSON structure for adding a new vector.
type AddRequest struct {
	ID     string    `json:"id,omitempty"`
	Vector []float64 `json:"vector"`
}

// SearchRequest defines the JSON structure for a search query.
type SearchRequest struct {
	Vector []float64 `json:"vector"`
	K      int       `json:"k"`
	Exact  bool      `json:"exact,omitempty"` // Use exact search instead of ANN
}

// API provides handlers for the HTTP server.
type API struct {
	db *HyperbolicDB
}

func (a *API) handleAdd(w http.ResponseWriter, r *http.Request) {
	var req AddRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if len(req.Vector) == 0 {
		http.Error(w, "Vector field is required", http.StatusBadRequest)
		return
	}

	item := Item{ID: req.ID, Vector: req.Vector}
	id, err := a.db.Add(item)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"id": id, "status": "added"})
}

func (a *API) handleSearch(w http.ResponseWriter, r *http.Request) {
	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if len(req.Vector) == 0 {
		http.Error(w, "Vector field is required", http.StatusBadRequest)
		return
	}
	if req.K <= 0 {
		http.Error(w, "K must be a positive integer", http.StatusBadRequest)
		return
	}

	var results []SearchResult
	var err error

	start := time.Now()
	if req.Exact {
		results, err = a.db.ExactSearch(req.Vector, req.K)
	} else {
		results, err = a.db.Search(req.Vector, req.K)
	}
	searchTime := time.Since(start)

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	method := "ann"
	if req.Exact {
		method = "exact"
	}

	response := map[string]interface{}{
		"results":     results,
		"search_time": searchTime.Nanoseconds() / 1000000,
		"method":      method,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (a *API) handleStats(w http.ResponseWriter, r *http.Request) {
	stats := a.db.Stats()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func main() {
	config := DefaultConfig()

	db, err := NewHyperbolicDB(config)
	if err != nil {
		log.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	api := &API{db: db}

	// Pre-populate with sample hierarchical data if database is empty
	stats := db.Stats()
	if stats["total_items"].(int) == 0 {
		log.Println("Populating database with sample data...")
		sampleData := []Item{
			{ID: "animal", Vector: Vector{0.0, 0.0}},
			{ID: "mammal", Vector: Vector{0.3, 0.2}},
			{ID: "reptile", Vector: Vector{-0.2, -0.4}},
			{ID: "bird", Vector: Vector{0.1, 0.5}},
			{ID: "dog", Vector: Vector{0.6, 0.3}},
			{ID: "cat", Vector: Vector{0.5, -0.1}},
			{ID: "snake", Vector: Vector{-0.4, -0.6}},
			{ID: "eagle", Vector: Vector{0.2, 0.7}},
			{ID: "golden_retriever", Vector: Vector{0.8, 0.35}},
			{ID: "poodle", Vector: Vector{0.75, 0.2}},
			{ID: "siamese_cat", Vector: Vector{0.7, -0.05}},
			{ID: "persian_cat", Vector: Vector{0.65, -0.15}},
		}

		for _, item := range sampleData {
			if _, err := db.Add(item); err != nil {
				log.Printf("Failed to add sample item %s: %v", item.ID, err)
			}
		}
		log.Printf("Added %d sample items", len(sampleData))
	}

	http.HandleFunc("/add", api.handleAdd)
	http.HandleFunc("/search", api.handleSearch)
	http.HandleFunc("/stats", api.handleStats)

	port := "8080"
	log.Printf("Starting Enhanced Hyperbolic Vector DB server on port %s...", port)
	log.Printf("Data directory: %s", config.DataPath)
	log.Printf("Endpoints:")
	log.Printf("  POST http://localhost:%s/add - Add vectors", port)
	log.Printf("  POST http://localhost:%s/search - Search vectors (ANN)", port)
	log.Printf("  GET  http://localhost:%s/stats - Database statistics", port)
	log.Printf("Database stats: %+v", db.Stats())

	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Failed to start server: %s", err)
	}
}
