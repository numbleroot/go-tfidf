package tfidf

import (
	"bytes"
	"math"
	"strings"

	"github.com/blevesearch/go-porterstemmer"
	"github.com/lytics/multibayes"
)

// Structs and types

type weightingScheme int

// Constants

const (

	// See: https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Definition

	// Term frequency weightings:
	// * Binary weighting.
	TermWeightingBinary weightingScheme = 0
	// * Raw frequency weighting.
	TermWeightingRaw weightingScheme = 1
	// * Log normalization weighting.
	TermWeightingLog weightingScheme = 2
	// * Double normalization 0.5 weighting.
	TermWeightingDoubleHalf weightingScheme = 3
	// * Double normalization K weighting.
	TermWeightingDoubleK weightingScheme = 4

	// Inverse document frequency weightings:
	// * Unary weighting.
	InvDocWeightingUnary weightingScheme = 0
	// * Log weighting.
	InvDocWeightingLog weightingScheme = 1
	// * Log smooth weighting.
	InvDocWeightingLogSmooth weightingScheme = 2
	// * Log maximum weighting.
	InvDocWeightingLogMax weightingScheme = 3
	// * Probabilistic weighting.
	InvDocWeightingProb weightingScheme = 4
)

var (
	c = multibayes.NewClassifier()
	t = c.Tokenizer
)

// Functions

// Takes an input document in string representation
// and tokenizes it. Along the way, stop bytes in the
// document will be removed and each term left will only
// find its way into the output list in its stemmed form.
//
// This function was heavily inspired by Allison Morgan's
// 'AddDocument' function from her 'tfidf' package:
// https://github.com/allisonmorgan/tfidf/blob/master/tfidf.go#L36
func TokenizeDocument(document string) []string {

	// Reserve space for result list (tokenized document).
	resultDocument := make([]string, 0)

	// Tokenize the supplied document.
	tokens := t.Tokenize([]byte(strings.ToLower(document)))

	// Range over all produced tokens.
	for _, token := range tokens {

		// Boolean signal whether to include or exclude one token.
		exclude := false

		// Range over all stop bytes from multibayes package
		// and remove each from tokens list of input document.
		for _, stopByte := range stopbytes {

			if bytes.Equal(token.Term, stopByte) {
				exclude = true
				break
			}
		}

		// Import iteration break: If token already considered,
		// leave current iteration here.
		if exclude {
			continue
		}

		// Alright, token is a new one. Stem and add it to result list.
		tokenStemmed := porterstemmer.StemString(string(token.Term))
		resultDocument = append(resultDocument, tokenStemmed)
	}

	// Return the tokenized document. Might be of len() = 0.
	return resultDocument
}

// This function calculates the number of occurencies of a given
// term in a given document. Based on the specified weighting scheme,
// the result value will be in a specific form. This functions
// expects a term, possibly stems it and looks up its frequency
// in an already tokenized document.
func TermFrequency(term string, stem bool, document []string, weighting weightingScheme) float64 {

	// Set frequency to 0 initially.
	var frequency float64
	frequency = 0.0

	if stem {
		// Stem input term.
		term = porterstemmer.StemString(term)
	}

	// Iterate over tokens in document.
	for _, token := range document {

		// If we find the term in the tokens,
		// increment frequency counter.
		if term == token {
			frequency += 1.0
		}
	}

	// Apply supplied weighting scheme.
	switch weighting {
	case TermWeightingLog:
		if frequency != 0.0 {
			// Apply log normalization.
			frequency = 1.0 + math.Log(frequency)
		}
	}

	return frequency
}

// This function takes in a compareDocument for which it will
// return the frequency of tokens in it. The number and order of
// tokens will be obtained by the given documents corpora.
// Note that compareDoc usually is in the corpora and both lists
// contain already tokenized elements.
func TermFrequencies(compareDoc []string, documents [][]string) []float64 {

	// Initialize result frequency vector and appearance map.
	frequencies := make([]float64, 0)
	appearance := make(map[string]bool)

	// Range over all documents.
	for _, document := range documents {

		// Range over all tokens in current document.
		for _, token := range document {

			// Check if we already considered this token.
			if exists := appearance[token]; !exists {

				// Add the frequency of the new token in compareDoc to vector.
				frequencies = append(frequencies, TermFrequency(token, false, compareDoc, TermWeightingRaw))

				// Set visited value for this token to true.
				appearance[token] = true
			}
		}
	}

	return frequencies
}

// Takes in a term, possibly stems it and counts its appearance
// in the supplied set of already tokenized documents. The resulting
// value will be altered by supplied weighting scheme.
func InverseDocumentFrequency(term string, stem bool, documents [][]string, weighting weightingScheme) float64 {

	// Declare result value.
	var idf float64

	if stem {
		// Stem input term.
		term = porterstemmer.StemString(term)
	}

	// Number of documents considered.
	numDocs := len(documents)

	// Number of documents in which supplied term is present.
	// To avoid a dision-by-zero, this term is initially set to zero.
	numDocsWithTerm := 1.0

	// Range over all documents.
	for _, document := range documents {

		contained := false

		// Range over all tokens in document.
		for i := 0; !contained && i < len(document); i++ {

			// If the current document contains our stemmed term,
			// increase the counter by one and leave the current document.
			if term == document[i] {
				numDocsWithTerm += 1.0
				contained = true
			}
		}
	}

	switch weighting {
	case InvDocWeightingLog:
		// Apply log on quotient.
		idf = math.Log(float64(numDocs) / numDocsWithTerm)
	}

	return idf
}

// Wrapper function to retrieve the map[string]float64 representation
// of an inverse document frequency vector for all terms in the supplied
// corpus, e.g. all tokenized documents.
func InverseDocumentFrequencies(documents [][]string, weighting weightingScheme) map[string]float64 {

	// Initialize result and appearance map.
	idfs := make(map[string]float64)
	appearance := make(map[string]bool)

	// Range over all documents.
	for _, document := range documents {

		// Range over all tokens in current document.
		for _, token := range document {

			// Check if we already considered this token.
			if exists := appearance[token]; !exists {

				// If we did not - add its idf value to the result map.
				idfs[token] = InverseDocumentFrequency(token, false, documents, weighting)

				// Set visited value for this token to true.
				appearance[token] = true
			}
		}
	}

	return idfs
}
