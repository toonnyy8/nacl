;;;; grad.lisp

(in-package #:nacl)

(defun nn/linear (in-features out-features &optional (bias t))
  (let ((params (lambda () 
                  `(,(t/randn `(,in-features ,out-features))
                    ,(t/randn `(,out-features)))))
        (fwfn (lambda (states x)
                (t/+ (t/matmul x (car states)) (car (last states))))))
   `(,params ,fwfn)))

(defun nn/fn (op)
  (let ((params (lambda () nil))
        (fwfn op))
    `(,params ,fwfn)))

(defun nn/seq (&rest layers)
  (let ((params (lambda ()
                  (loop for layer in layers
                        collect (funcall (car layer)))))
        (fwfn (lambda (states x)
                (let ((inp x))
                  (loop for layer in layers
                        for state in states
                        do (setf inp (funcall (car (last layer)) state inp)))
                  inp))))
    `(,params ,fwfn)))

(let* ((model (nn/linear 2 32))
       (params (car model))
       (states (funcall params))
       (fwfn (car (last model)))
       (x (t/randn '(16 2))))
  (t/shape (funcall fwfn states x)))

(last '(1 2 3))

(let* ((model (nn/seq (nn/linear 2 32)
                      (nn/linear 32 32)
                      (nn/linear 32 32)
                      (nn/linear 32 3)))
       (params (car model))
       (states (funcall params))
       (fwfn (car (last model)))
       (x (t/randn '(16 2))))
  (t/shape (funcall fwfn states x)))
(t/data (car (funcall (car (nn/linear 5 5)))))
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
