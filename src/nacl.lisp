;;;; NaCL.lisp

(in-package #:nacl)
;; 
;; (defpackage #:nacl/t
;;   (:use #:cl)
;;   (:export #:new
;;            #:data
;;            #:grad
;;            #:bwfn
;;            #:+
;; 
;; (defpackage #:nacl/t/bw
;;   (:use #:cl)
;;   (:export #:+)


;; (defun add (a b) 
;;   (let* ((data (numcl:+ (data a) (data b)))) 
;;     (make-instance
;;       'tensor
;;       :data data
;;       :shape shape
;;       :grad (numcl:zeros shape)
;;       :chain (lambda (grad) 1))
;; 
;; (defun sub (a b) (numcl:- (data a) (data b)))
;; 
;; (defun mul (a b) (numcl:* (data a) (data b)))
;; 
;; (defun div (a b) (numcl:/ (data a) (data b)))
;; 
;; (defun matmul (a b) (numcl:matmul (data a) (data b)))
