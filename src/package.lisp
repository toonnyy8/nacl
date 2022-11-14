;;;; package.lisp

;; (defpackage #:nacl
;;   (:use #:cl)
;;   (:export #:add
;;            #:sub
;;            #:mul
;;            #:div
;;            #:matmul
;;            #:t/ones
;;            #:t/data
;;            #:t/shape
;;            #:t/grad
;;            #:t/zeros
;;            #:t/randn


;; (defpackage #:nacl
;;   (:use #:cl)
;;   (:export #:nacl/t
;;            #:nacl/t/bw

 (defpackage #:nacl
   (:use #:cl)
   (:export #:t/new
            #:t/data
            #:t/tid
            #:t/bwfn
            #:t/+
            #:t/-
            #:t/*
            #:t/div
            #:t/expt
            #:t/log
            #:t/bw))

;; (in-package :cl-user)
;; 
;; (defpackage #:nacl/t.exported
;;   (:use)
;;   (:export #:new
;;            #:data
;;            #:grad
;;            #:bwfn
;;            #:+
;; 
;; (uiop:define-package #:nacl/t.impl
;;   (:mix #:cl                           
;;         #:nacl/t.exported
;; 
;; (uiop:define-package #:nacl/t
;;   (:mix #:nacl/t.exported #:cl)
;;   (:reexport #:nacl/t.exported)
;;   (:reexport #:cl)
;; 
;; 
;; (defpackage #:nacl/t/bw.exported
;;   (:use)
;;   (:export #:+)
;; 
;; (uiop:define-package #:nacl/t/bw.impl
;;   (:mix #:cl                           
;;         #:nacl/t/bw.exported
;; 
;; (uiop:define-package #:nacl/t/bw
;;   (:mix #:nacl/t/bw.exported #:cl)
;;   (:reexport #:nacl/t/bw.exported)
;;   (:reexport #:cl)

;; (defpackage #:nacl/t
;;   ;; (:use #:cl)
;;   (:mix #:cl)
;;   (:export #:new
;;            #:data
;;            #:grad
;;            #:bwfn
;;            #:+))

;; (defpackage #:nacl/t/bw
;;   ;; (:use #:cl)
;;   (:mix #:cl)
;;   (:export #:+))
