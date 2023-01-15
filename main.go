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
	Rnd    *rand.Rand
	Set    tf32.Set
	Input  *tf32.V
	L1     tf32.Meta
	Cost   tf32.Meta
	I      float64
	Points plotter.XYs
}

// NewEntropyLayer creates a new entropy layer
func NewEntropyLayer() *EntropyLayer {
	rnd := rand.New(rand.NewSource(1))

	others := tf32.NewSet()
	others.Add("inputs", 2, 1)
	inputs := others.ByName["inputs"]
	inputs.X = inputs.X[:cap(inputs.X)]

	// Create the weight data matrix
	set := tf32.NewSet()
	set.Add("w1", 2, 3)
	set.Add("b1", 3, 1)
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

	// The neural network is the attention model from attention is all you need
	l1 := tf32.ReLu(tf32.Add(tf32.Mul(set.Get("w1"), others.Get("inputs")), set.Get("b1")))
	cost := tf32.Entropy(tf32.Softmax(tf32.T(tf32.Mul(tf32.Softmax(l1), tf32.T(set.Get("w1"))))))

	return &EntropyLayer{
		Rnd:    rnd,
		Set:    set,
		Input:  inputs,
		L1:     l1,
		Cost:   cost,
		Points: make(plotter.XYs, 0, 8),
	}
}

// Step steps the layer forward
func (e *EntropyLayer) Step(sign float64, in []float32) float32 {
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
			e.Set.Weights[j].X[k] += float32(sign) * Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
		}
	}

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

	err = p.Save(8*vg.Inch, 8*vg.Inch, "entropy_cost.png")
	if err != nil {
		panic(err)
	}

	e.Set.Save("set.w", 0, 0)
}

func main() {
	rnd, sign, inputs := rand.New(rand.NewSource(1)), -1.0, make([]float32, 2)
	entropy := NewEntropyLayer()
	// The stochastic gradient descent loop
	for i := 0; i < 64; i++ {
		if i&1 == 0 {
			sign = -1
			inputs[0] = float32(rnd.Intn(2))
			inputs[1] = float32(rnd.Intn(2))
		} else {
			sign = .5
			inputs[0] = float32(.5 + rnd.NormFloat64())
			inputs[1] = float32(.5 + rnd.NormFloat64())
		}

		start := time.Now()
		// Step the model
		loss := entropy.Step(sign, inputs)

		end := time.Since(start)
		fmt.Println(i, loss, end)

		if math.IsNaN(float64(loss)) {
			fmt.Println(loss)
			break
		}
	}

	data := [][]float32{
		{0, 0, 0},
		{0, 1, 1},
		{1, 0, 1},
		{1, 1, 0},
	}
	for _, example := range data {
		entropy.Input.X[0] = example[0]
		entropy.Input.X[1] = example[1]
		entropy.L1(func(a *tf32.V) bool {
			fmt.Println(example, a.X)
			return true
		})
	}

	entropy.Save()
}
