;;;; grad.lisp

(in-package #:nacl)

(defun nn/linear (in-features out-features &optional (bias t))
  (let ((initfn (lambda () 
                  `(,(t/randn `(,in-features ,out-features) 
                              :std (/ 1 (sqrt in-features)))
                    ,(t/randn `(,out-features)))))
        (fwfn (lambda (states x)
                (let* ((w (car states))
                       (b (nth 1 states))
                       (n (length (t/shape x)))
                       (b (t/reshape b (loop for i from 0 to (- n 1)
                                             collect (if (= i (- n 1)) 
                                                         out-features 1)))))
                  (t/+ (t/matmul x w) b)))))
   `(,initfn ,fwfn)))

(defun nn/fn (op)
  (let ((initfn (lambda () nil))
        (fwfn (lambda (states x) (funcall op x))))
    `(,initfn ,fwfn)))

(defun nn/seq (&rest layers)
  (let ((initfn (lambda ()
                  (loop for layer in layers
                        collect (funcall (car layer)))))
        (fwfn (lambda (states x)
                (let ((inp x))
                  (loop for layer in layers
                        for state in states
                        do (setf inp (funcall (nth 1 layer) state inp)))
                  inp))))
    `(,initfn ,fwfn)))


(defun nn/u/flatten-states (nn-states)
  (if (typep nn-states 'list)
      (apply #'concatenate 
             (concatenate 'list '(list) 
               (loop for nn-state in nn-states
                     collect (nn/u/flatten-states nn-state))))
      (if (typep nn-states 'tensor)
          `(,nn-states) nil)))

(defun nn/u/shaping-states (nn-states flattened-states)
  (if (typep nn-states 'list)
      (values (loop for nn-state in nn-states
                    collect (let ((state nil)) 
                              (setf (values state flattened-states) 
                                    (nn/u/shaping-states nn-state 
                                                         flattened-states))
                              state))
              flattened-states)
      (if (typep nn-states 'tensor)
          (values (car flattened-states) (cdr flattened-states))
          (values nn-states flattened-states))))

;; (let* ((model (nn/linear 2 32))
;;        (initfn (car model))
;;        (states (funcall initfn))
;;        (fwfn (nth 1 model))
;;        (x (t/randn '(16 2))))
;;   (t/shape (funcall fwfn states x)))
;; 
;; (let* ((model (nn/seq (nn/linear 2 32)
;;                       (nn/fn #'f/relu)
;;                       (nn/linear 32 32)
;;                       (nn/fn #'f/relu)
;;                       (nn/linear 32 32)
;;                       (nn/fn #'f/relu)
;;                       (nn/linear 32 3)))
;;        (initfn (car model))
;;        (states (funcall initfn))
;;        (fwfn (nth 1 model))
;;        (x (t/randn '(16 2))))
;;   (print states)
;;   (print (nn/u/flatten-states states))
;;   (print (nn/u/shaping-states 
;;           states 
;;           (mapcar #'t/zeros-like (nn/u/flatten-states states))))
;;   (t/shape (funcall fwfn states x)))

;; (t/data (car (funcall (car (nn/linear 5 5)))))

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
