;;;; package.lisp

(defpackage #:nacl
  (:use #:cl)
  (:import-from :numcl)
  (:export #:t/shape
           #:t/new
           #:t/full
           #:t/full-like
           #:t/randn
           #:t/randn-like
           #:t/ones
           #:t/ones-like
           #:t/zeros
           #:t/zeros-like
           #:t/data
           #:t/tid
           #:t/bwfn
           #:t/+
           #:t/-
           #:t/*
           #:t/div
           #:t/expt
           #:t/exp
           #:t/log
           #:t/matmul
           #:t/reshape
           #:t/sum
           #:t/mean
           #:t/cos
           #:t/sin
           #:t/stack
           #:t/unstack
           #:t/cat
           #:t/split
           #:t/bw
           #:t/sg
           #:t/gcp
           #:f/relu
           #:f/sigmoid
           #:f/bce
           #:nn/linear
           #:nn/fn
           #:nn/seq
           #:optim/adam
           #:optim/sgd))
