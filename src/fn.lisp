;;;; fn.lisp

(in-package #:nacl)

(defun f/relu (x)
  (let* ((mask (numcl:>= (t/data x) (numcl:zeros-like (t/data x))))
         (mask (t/new mask)))
    (t/* x mask)))

(defun f/sigmoid (x)
  (t/gcp (t/div 1 (t/+ 1 (t/exp x)))))

(defun f/bce (pred tgt)
  (t/- (t/+ (t/* tgt (t/log pred)) 
            (t/* (t/- 1 tgt) (t/log (t/- 1 pred))))))
  
