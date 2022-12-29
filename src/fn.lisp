;;;; fn.lisp

(in-package #:nacl)

(defun f/relu (x)
  (let* ((mask (numcl:>= (t/data x) (numcl:zeros-like (t/data x))))
         (mask (t/new mask)))
    (t/* x mask)))
