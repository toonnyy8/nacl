;;;; grad.lisp

(in-package #:nacl)

(defun nn/linear (in-features out-features &optional (bias t))
  (let ((params (lambda () 
                  `(,(t/randn `(,in-features ,out-features))
                    ,(t/randn `(,out-features)))))
        (fwfn (lambda (states x)
                (t/matmul x (car x)))))
   `(,params ,fwfn)))

(defun nn/fn (op)
  (let ((params (lambda () nil))
        (fwfn op)))
  `(,params ,fwfn))
;; (funcall (car (nn/linear 5 5)))

;; (defun t/u/matmul (x y)
;;  (let* ((x-shape (numcl:shape x))
;;         (y-shape (numcl:shape y))
;;         (x-dims (length x-shape))
;;         (y-dims (length y-shape))
;;         (max-dims (max x-dims y-dims))
;;         (x-shape (concatenate 'list 
;;                               (loop repeat (- max-dims x-dims) collect 1)
;;                               x-shape '(1)))
;;         (y-shape (concatenate 'list 
;;                               (loop repeat (+ 1 (- max-dims y-dims)) collect 1)
;;                               y-shape))
;;         (new-x (numcl:reshape x x-shape))
;;         (new-y (numcl:reshape y y-shape)))
;;    (numcl:sum (numcl:* new-x new-y) :axes (- max-dims 1))))
;; 
;; (let ((x (numcl:ones '(2 1)))
;;       (y (numcl:ones '(1 2))))
;;   (t/u/matmul x y))
