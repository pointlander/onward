// Copyright 2023 The Onward Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.9
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.999
	// Eta is the learning rate
	Eta = .3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// EntropyLayer is an auto learning layer
type EntropyLayer struct {
	Name   string
	Rnd    *rand.Rand
	Set    tf32.Set
	Others tf32.Set
	Input  *tf32.V
	L1     tf32.Meta
	Cost   tf32.Meta
	I      float64
	Points plotter.XYs
}

// NewEntropyLayer creates a new entropy layer
func NewEntropyLayer(name string, inputSize, outputSize, batchSize int, weights []float32) *EntropyLayer {
	rnd := rand.New(rand.NewSource(1))

	others := tf32.NewSet()
	others.Add("inputs", inputSize, batchSize)
	inputs := others.ByName["inputs"]
	inputs.X = inputs.X[:cap(inputs.X)]

	// Create the weight data matrix
	set := tf32.NewSet()
	set.Add("w1", inputSize, outputSize)
	set.Add("b1", outputSize, 1)
	for _, w := range set.Weights {
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float32, len(w.X))
			}
			continue
		}
		if len(weights) == 0 {
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, float32(rnd.NormFloat64()*factor))
			}
		} else {
			w.X = w.X[:cap(w.X)]
			copy(w.X, weights)
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}

	// The neural network is the attention model from attention is all you need
	x := tf32.Add(tf32.Mul(set.Get("w1"), others.Get("inputs")), set.Get("b1"))
	l1 := tf32.Everett(x)
	cost := tf32.Sum(tf32.Entropy(tf32.Softmax(tf32.T(tf32.Mul(tf32.Softmax(x), tf32.T(set.Get("w1")))))))

	return &EntropyLayer{
		Name:   name,
		Rnd:    rnd,
		Set:    set,
		Others: others,
		Input:  inputs,
		L1:     l1,
		Cost:   cost,
		Points: make(plotter.XYs, 0, 8),
	}
}

// Step steps the layer forward
func (e *EntropyLayer) Step(in []float32) float32 {
	e.I++
	i := e.I
	copy(e.Input.X, in)
	loss := tf32.Gradient(e.Cost).X[0]

	// Update the point weights with the partial derivatives using adam
	b1, b2 := float32(math.Pow(B1, i)), float32(math.Pow(float64(B2), i))
	for j, w := range e.Set.Weights {
		for k, d := range w.D {
			g := d
			m := B1*w.States[StateM][k] + (1-B1)*g
			v := B2*w.States[StateV][k] + (1-B2)*g*g
			w.States[StateM][k] = m
			w.States[StateV][k] = v
			mhat := m / (1 - b1)
			vhat := v / (1 - b2)
			e.Set.Weights[j].X[k] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
		}
	}

	e.Set.Zero()
	e.Others.Zero()
	e.Points = append(e.Points, plotter.XY{X: i, Y: float64(loss)})

	return loss
}

// Save plots and saves the model
func (e *EntropyLayer) Save() {
	// Plot the cost
	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(e.Points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%s_entropy_cost.png", e.Name))
	if err != nil {
		panic(err)
	}

	e.Set.Save(fmt.Sprintf("%s_entropy_set.w", e.Name), 0, 0)
}

// SupervisedLayer is an supervised learning layer
type SupervisedyLayer struct {
	Name    string
	Rnd     *rand.Rand
	Set     tf32.Set
	Others  tf32.Set
	Input   *tf32.V
	Targets *tf32.V
	L1      tf32.Meta
	Cost    tf32.Meta
	I       float64
	Points  plotter.XYs
}

func log(a float32) float32 {
	return float32(math.Log(float64(a)))
}

// CrossEntropy computes the cross entropy cost of two tensors
func CrossEntropy(k tf32.Continuation, node int, a, b *tf32.V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, size := tf32.NewV(a.S[1]), len(a.X)

	for i := 0; i < size; i += width {
		av, bv, sum := a.X[i:i+width], b.X[i:i+width], float32(0.0)
		for j, ax := range av {
			bx := bv[j]
			if bx == 1 {
				sum += log(ax + .000001)
			} else {
				sum += log(1 - ax + .000001)
			}
		}
		c.X = append(c.X, -sum)
	}
	if k(&c) {
		return true
	}
	index := 0
	for i := 0; i < size; i += width {
		av, bv, ad, bd, d := a.X[i:i+width], b.X[i:i+width], a.D[i:i+width], b.D[i:i+width], c.D[index]
		for j, ax := range av {
			bx := bv[j]
			if bx == 1 {
				ad[j] -= d / (ax + .000001)
				bd[j] -= log(ax+.000001) * d
			} else {
				ad[j] += d / (1 - ax + .000001)
				bd[j] -= log(1-ax+.000001) * d
			}
		}
		index++
	}
	return false
}

// NewEntropyLayer creates a new entropy layer
func NewSupervisedLayer(name string, inputSize, outputSize, batchSize int,
	activation func(a tf32.Meta, options ...map[string]interface{}) tf32.Meta,
	loss func(a, b tf32.Meta, options ...map[string]interface{}) tf32.Meta) *SupervisedyLayer {
	rnd := rand.New(rand.NewSource(1))

	others := tf32.NewSet()
	others.Add("inputs", inputSize, batchSize)
	others.Add("targets", outputSize, batchSize)
	inputs := others.ByName["inputs"]
	inputs.X = inputs.X[:cap(inputs.X)]
	targets := others.ByName["targets"]
	targets.X = targets.X[:cap(targets.X)]

	// Create the weight data matrix
	set := tf32.NewSet()
	set.Add("w1", inputSize, outputSize)
	set.Add("b1", outputSize, 1)
	for _, w := range set.Weights {
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float32, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rnd.NormFloat64()*factor))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}

	l1 := activation(tf32.Add(tf32.Mul(set.Get("w1"), others.Get("inputs")), set.Get("b1")))
	cost := tf32.Sum(loss(l1, others.Get("targets")))

	return &SupervisedyLayer{
		Name:    name,
		Rnd:     rnd,
		Set:     set,
		Others:  others,
		Input:   inputs,
		Targets: targets,
		L1:      l1,
		Cost:    cost,
		Points:  make(plotter.XYs, 0, 8),
	}
}

// Step steps the layer forward
func (s *SupervisedyLayer) Step(in, targets []float32) float32 {
	s.I++
	i := s.I
	copy(s.Input.X, in)
	copy(s.Targets.X, targets)
	loss := tf32.Gradient(s.Cost).X[0]

	// Update the point weights with the partial derivatives using adam
	b1, b2 := float32(math.Pow(B1, i)), float32(math.Pow(float64(B2), i))
	for j, w := range s.Set.Weights {
		for k, d := range w.D {
			g := d
			m := B1*w.States[StateM][k] + (1-B1)*g
			v := B2*w.States[StateV][k] + (1-B2)*g*g
			w.States[StateM][k] = m
			w.States[StateV][k] = v
			mhat := m / (1 - b1)
			vhat := v / (1 - b2)
			s.Set.Weights[j].X[k] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
		}
	}

	s.Set.Zero()
	s.Others.Zero()
	s.Points = append(s.Points, plotter.XY{X: i, Y: float64(loss)})

	return loss
}

// Save plots and saves the model
func (s *SupervisedyLayer) Save() {
	// Plot the cost
	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(s.Points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%s_supervised_cost.png", s.Name))
	if err != nil {
		panic(err)
	}

	s.Set.Save(fmt.Sprintf("%s_supervised_set.w", s.Name), 0, 0)
}

// XORExample is an example of the XOR problem
func XORExample() {
	inputs := []float32{-1, -1, -1, 1, 1, -1, 1, 1}
	targets := []float32{-1, 1, 1, -1}
	entropy := NewEntropyLayer("xor", 2, 4, 4, inputs)
	supervised := NewSupervisedLayer("xor", 2*4, 1, 4, tf32.TanH, tf32.Quadratic)

	// The stochastic gradient descent loop
	for i := 0; i < 64; i++ {
		start := time.Now()
		// Step the model
		var loss float32
		loss = entropy.Step(inputs)
		var next *tf32.V
		entropy.L1(func(a *tf32.V) bool {
			next = a
			return true
		})
		loss += supervised.Step(next.X, targets)
		end := time.Since(start)
		fmt.Println(i, loss, end)

		if math.IsNaN(float64(loss)) {
			fmt.Println(loss)
			break
		}
	}

	copy(entropy.Input.X, inputs)
	entropy.L1(func(a *tf32.V) bool {
		copy(supervised.Input.X, a.X)
		fmt.Println(a.X)
		supervised.L1(func(a *tf32.V) bool {
			fmt.Println(a.X)
			if a.X[0] > 0 {
				panic("should be -1")
			}
			if a.X[1] < 0 {
				panic("should be 1")
			}
			if a.X[2] < 0 {
				panic("should be 1")
			}
			if a.X[3] > 0 {
				panic("should be -1")
			}
			return true
		})
		return true
	})

	entropy.Save()
	supervised.Save()
}

// IRISExample is an example of the IRIS problem
func IRISExample() {
	rnd := rand.New(rand.NewSource(1))

	// Load the iris data set
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	fisher := datum.Fisher
	length := len(fisher)
	for _, value := range fisher {
		sum := 0.0
		for _, measure := range value.Measures {
			sum += measure * measure
		}
		sum = math.Sqrt(sum)
		for i := range value.Measures {
			value.Measures[i] /= sum
		}
	}

	inputs := make([]float32, 0, length*4)
	targets := make([]float32, length*3)
	for i, item := range fisher {
		measures := item.Measures
		inputs = append(inputs, float32(measures[0]), float32(measures[1]), float32(measures[2]), float32(measures[3]))
		targets[i*3+iris.Labels[item.Label]] = 1
	}
	crossEntropy := tf32.B(CrossEntropy)
	entropy := NewEntropyLayer("iris", 4, 8, 1, nil)
	supervised := NewSupervisedLayer("iris", 2*8, 3, 1, tf32.Softmax, crossEntropy)

	// The stochastic gradient descent loop
	for i := 0; i < 2048*1024; i++ {
		start := time.Now()
		index := rnd.Intn(length)
		// Step the model
		var loss float32
		loss = entropy.Step(inputs[index*4 : index*4+4])
		var next *tf32.V
		entropy.L1(func(a *tf32.V) bool {
			next = a
			sum := float32(0.0)
			for _, value := range next.X {
				sum += value * value
			}
			scale := float32(math.Sqrt(float64(sum)))
			for i, value := range next.X {
				next.X[i] = value / scale
			}
			return true
		})
		loss += supervised.Step(next.X, targets[index*3:index*3+3])
		end := time.Since(start)
		fmt.Println(i, loss, end)

		if math.IsNaN(float64(loss)) {
			fmt.Println(loss)
			break
		}
	}

	correct := 0
	for i := 0; i < length; i++ {
		copy(entropy.Input.X, inputs[i*4:i*4+4])
		entropy.L1(func(a *tf32.V) bool {
			sum := float32(0.0)
			for _, value := range a.X {
				sum += value * value
			}
			scale := float32(math.Sqrt(float64(sum)))
			for i, value := range a.X {
				a.X[i] = value / scale
			}
			copy(supervised.Input.X, a.X)
			supervised.L1(func(a *tf32.V) bool {
				index, max := 0, float32(0.0)
				for i, value := range a.X {
					if value > max {
						index, max = i, value
					}
				}
				target := targets[i*3 : i*3+3]
				fmt.Println(i, index, target)
				if target[index] == 1.0 {
					correct++
				}
				return true
			})
			return true
		})
	}
	fmt.Println("correct=", correct, float64(correct)/float64(length))
	entropy.Save()
	supervised.Save()
}

func main() {
	XORExample()
	IRISExample()
}
