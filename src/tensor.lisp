;;;; tensor.lisp

(in-package nacl)

(defclass tensor ()
  ((data
    :initarg :data
    :accessor data)
   (bwfn
    :initarg :bwfn
    :accessor bwfn)))

(setf (fdefinition 't/data) #'data)
(setf (fdefinition 't/bwfn) #'bwfn)

(defun t/shape (x)
  (check-type x tensor)
  (numcl:shape (t/data x)))

(defstruct var-grad-tuple
  (v nil :type tensor)
  (g nil :type tensor))

(defmacro t/prototype-op (fw bw)
  `(let ((tmp (make-instance
               'tensor
               :data ,fw
               :bwfn nil)))
     (setf (bwfn tmp) ,bw)
     tmp))

(defmacro t/init-bw () `(lambda (j) (list (make-var-grad-tuple :v tmp :g j))))

(defun t/new (array) (t/prototype-op (numcl:asarray array) (t/init-bw)))

(defun t/ones (shape) (t/prototype-op (numcl:ones shape) (t/init-bw)))

(defun t/ones-like (array) 
  (t/prototype-op (numcl:ones-like (data array)) (t/init-bw)))

(defun t/zeros (shape) (t/prototype-op (numcl:zeros shape) (t/init-bw)))

(defun t/zeros-like (array) 
  (t/prototype-op (numcl:zeros-like (data array)) (t/init-bw)))

(defun t/randn (shape &key (mean 0.d0) (std 1.d0))
  (t/prototype-op (numcl:normal (coerce mean 'double-float)
                                (coerce std 'double-float) shape)
                  (t/init-bw)))

(defun t/randn-like (array &key (mean 0.d0) (std 1.d0))
  (t/prototype-op (numcl:normal (coerce mean 'double-float)
                                (coerce std 'double-float) 
                                (numcl:shape (data array)))
                  (t/init-bw)))

(defun t/rand (shape &key (low 0.d0) (high 1.d0))
  (t/prototype-op (numcl:uniform low high shape)
                  (t/init-bw)))

(defun t/rand-like (array &key (low 0.d0) (high 1.d0))
  (t/prototype-op (numcl:uniform low high (numcl:shape (data array)))
                  (t/init-bw)))

(defun t/full (shape value)
  (t/prototype-op (numcl:full shape value) (t/init-bw)))

(defun t/full-like (array value)
  (t/prototype-op (numcl:full (numcl:shape (data array)) value) (t/init-bw)))

(defun t/bw (out inps &optional j)
  (if (not j) (setq j (t/ones-like out)))
  (let ((gs (funcall (bwfn out) j))
        (*inps (nn/u/flatten-states inps)))
    (nn/u/shaping-states inps
      (mapcar (lambda (inp) 
                (reduce (lambda (prev vg)
                          (if (eq inp (var-grad-tuple-v vg))
                              (t/+ prev (var-grad-tuple-g vg))
                              prev))
                        gs :initial-value (t/zeros-like inp)))
              *inps))))

(defun t/sg (xs)
  (let ((*xs (nn/u/flatten-states xs)))
    (nn/u/shaping-states 
     xs (loop for x in *xs
              collect (t/new (t/data x))))))

(defun t/gcp (xs)
  (let ((*xs (nn/u/flatten-states xs)))
    (nn/u/shaping-states 
     xs (loop for x in *xs
              collect (let ((tmp (t/new (t/data x))))
                        (setf (bwfn tmp) (bwfn x))
                        tmp)))))

(defmacro t/op-bw (bw-op-name &rest args)
  `(lambda (j) (concatenate 'list 
                            (list (make-var-grad-tuple :v tmp :g j))
                            ,(concatenate 'list (list bw-op-name) args '(j)))))

;; (defun t/u/matmul (x y)
;;   (let* ((x-shape (numcl:shape x))
;;          (y-shape (numcl:shape y))
;;          (x-dims (length x-shape))
;;          (y-dims (length y-shape))
;;          (max-dims (max x-dims y-dims))
;;          (x-shape (concatenate 'list 
;;                                (loop repeat (- max-dims x-dims) collect 1)
;;                                x-shape '(1)))
;;          (y-shape (concatenate 'list 
;;                                (loop repeat (+ 1 (- max-dims y-dims)) collect 1)
;;                                y-shape))
;;          (new-x (numcl:reshape x x-shape))
;;          (new-y (numcl:reshape y y-shape))
;;     (numcl:sum (numcl:* new-x new-y) :axes (- max-dims 1))))

(defun t/matmul (x y)
  (let* ((x-shape (t/shape x))
         (y-shape (t/shape y))
         (x-dims (length x-shape))
         (y-dims (length y-shape))
         (max-dims (max x-dims y-dims))
         (new-x-shape 
           (concatenate 'list (loop repeat (- max-dims x-dims) collect 1)
                        x-shape '(1)))
         (new-y-shape 
           (concatenate 'list 
                        (loop repeat (+ 1 (- max-dims y-dims)) collect 1)
                        y-shape))
         (new-x (t/reshape x new-x-shape))
         (new-y (t/reshape y new-y-shape)))
    (t/sum (t/* new-x new-y) `(,(- max-dims 1)))))


;; (defun t/matmul (x y)
;;   (t/prototype-op (t/u/matmul (data x) (data y))
;;                   (t/op-bw t/bw/matmul x y))))

(defun t/T (x)
  (t/prototype-op (numcl:transpose (data x))
                  (t/op-bw t/bw/T x)))

(defun t/+ (x y)
  (let ((x (if (numberp x) (t/new `(,x)) x))
        (y (if (numberp y) (t/new `(,y)) y)))
    (t/prototype-op (numcl:+ (data x) (data y))
                    (t/op-bw t/bw/+ x y))))

(numberp nil)

(defun t/- (x &optional y)
  (let ((x (if (numberp x) (t/new `(,x)) x))
        (y (if (numberp y) (t/new `(,y)) y)))
    (if y
         (t/prototype-op (numcl:- (data x) (data y))
                     (t/op-bw t/bw/- x y))
         (t/prototype-op (numcl:- (data x))
                     (t/op-bw t/bw/neg x)))))

(defun t/* (x y)
  (let ((x (if (numberp x) (t/new `(,x)) x))
        (y (if (numberp y) (t/new `(,y)) y)))
    (t/prototype-op (numcl:* (data x) (data y))
                    (t/op-bw t/bw/* x y))))

(defun t/div (x y)
  (let ((x (if (numberp x) (t/new `(,x)) x))
        (y (if (numberp y) (t/new `(,y)) y)))
    (t/prototype-op (numcl:/ (data x) (data y))
                    (t/op-bw t/bw/div x y))))

(defun t/exp (x)
  (let ((x (if (numberp x) (t/new `(,x)) x)))
    (t/prototype-op (numcl:exp (data x))
                    (t/op-bw t/bw/exp x))))

(defun t/expt (x y)
  (let ((x (if (numberp x) (t/new `(,x)) x))
        (y (if (numberp y) (t/new `(,y)) y)))
    (t/prototype-op (numcl:expt (data x) (data y))
                    (t/op-bw t/bw/expt x y))))

(defun t/u/log (x &optional y)
  (numcl:/ (numcl:log x) (numcl:log y)))

(defun t/log (x &optional y)
  (let ((x (if (numberp x) (t/new `(,x)) x))
        (y (if (numberp y) (t/new `(,y))
               (if (not y) (t/new `(,(coerce (exp 1) 'double-float))) y))))
    (t/prototype-op (t/u/log (data x) (data y))
                    (t/op-bw t/bw/log x y))))

(defun t/reshape (x shape)
  (t/prototype-op (numcl:reshape (data x) shape)
                  (t/op-bw t/bw/reshape x)))

;; (defun t/u/reduce-shape (x axes) 
;;   (let ((org-shape (numcl:shape x)))
;;     (loop for dim in org-shape
;;           for axis from 0 to (length org-shape)
;;           collect (if (member axis axes) 1 dim)))

(defun t/u/reduce-shape (x axes) 
  (let ((org-shape (numcl:shape x)))
    (if axes
        (loop for dim in org-shape
              for axis from 0 to (- (length org-shape) 1)
              collect (if (member axis axes) 1 dim))
        (loop for dim in org-shape collect 1))))

;; (defun t/u/sum (x &optional (axes nil) &key (keepdim nil))
;;   (check-type x numcl:array)
;;   (let ((y (numcl:sum x :axes axes)))
;;     (if keepdim
;;         (let ((new-shape (t/u/reduce-shape x axes)))
;;           (if (numcl:arrayp y)
;;               (numcl:reshape y new-shape) 
;;               (numcl:full new-shape y)
;;         y)))

(defun t/u/sum (x &optional (axes nil) &key (keepdim nil))
  (check-type x numcl:array)
  (let* ((y (numcl:sum x :axes axes))
         (y (if (numcl:arrayp y) y 
                (if (float-features:float-nan-p y)
                    (numcl:asarray `(,0))
                    (numcl:asarray `(,y))))))
    (if keepdim (numcl:reshape y (t/u/reduce-shape x axes)) y)))

(defun t/sum (x &optional (axes nil) &key (keepdim nil))
  (t/prototype-op (t/u/sum (data x) axes :keepdim keepdim)
                  (t/op-bw t/bw/sum x (t/u/reduce-shape (data x) axes))))

(defun t/u/mean (x &optional (axes nil) &key (keepdim nil))
  (check-type x numcl:array)
  (let* ((y (numcl:mean x :axes axes))
         (y (if (numcl:arrayp y) y 
                (if (float-features:float-nan-p y)
                    (numcl:asarray `(,0))
                    (numcl:asarray `(,y))))))
    (if keepdim (numcl:reshape y (t/u/reduce-shape x axes)) y)))

(defun t/mean (x &optional (axes nil) &key (keepdim nil))
  (t/prototype-op (t/u/mean (data x) axes :keepdim keepdim)
                  (t/op-bw t/bw/mean x (t/u/reduce-shape (data x) axes))))

(defun t/cos (x)
  (t/prototype-op (numcl:cos (data x))
                  (t/op-bw t/bw/cos x)))

(defun t/sin (x)
  (t/prototype-op (numcl:sin (data x))
                  (t/op-bw t/bw/sin x)))

(defun t/stack (xs &optional (axis 0))
  (t/prototype-op (numcl:stack (mapcar #'data xs) :axis axis)
                  (t/op-bw t/bw/stack xs axis)))

(defun t/u/gen-mask (xs merge-fn)
  (let ((1masks (mapcar #'numcl:ones-like xs))
        (0masks (mapcar #'numcl:zeros-like xs))
        (n (- (length xs) 1)))
    (loop for i from 0 to n
          collect 
          (funcall 
           merge-fn 
           (loop for j from 0 to n
                 collect (nth j (if (= i j) 1masks 0masks)))))))

;; (let ((a (numcl:ones '(2 3))))
;;   (t/u/gen-mask `(,a ,a ,a) (lambda (xs) (numcl:stack xs))))

(defun t/unstack (x &optional (axis 0))
  (let* ((_xs (numcl:unstack (data x) :axis axis))
         (_pads (mapcar #'numcl:zeros-like _xs))
         (pads (mapcar #'t/new _pads))
         (n (length _xs)))
    (loop for i from 0 to (- n 1)
          collect (let ((idx i)
                        (_x (nth i _xs)))
                    (t/prototype-op 
                     _x (t/op-bw t/bw/unstack x axis idx pads))))))

;; (defun t/unstack (x &optional (axis 0))
;;   (let* ((_xs (numcl:unstack (data x) :axis axis))
;;          (_masks (t/u/gen-mask 
;;                   _xs (lambda (ms) (numcl:stack ms :axis axis)))
;;     (loop for _x in _xs
;;           for _mask in _masks
;;           collect (let ((mask (t/new _mask)))
;;                     (t/prototype-op 
;;                      _x (t/op-bw t/bw/unstack x mask))))))

(defun t/cat (xs &optional (axis 0))
  (t/prototype-op (numcl:concatenate (mapcar #'data xs) :axis axis)
                  (t/op-bw t/bw/cat xs axis)))

(defun t/u/split (x sections &optional (axis 0))
  (let* ((n (length (numcl:shape x)))
         (begin 0)
         (end 0)
         (subscripts (loop for section in sections
                           do (setf begin end)
                           do (setf end (+ end section))
                           collect (loop for i from 0 to (- n 1)
                                         collect (if (= i axis) 
                                                     `(,begin ,end) t)))))
    (loop for subscript in subscripts
          collect (apply #'numcl:aref 
                         (concatenate 'list `(,x) subscript)))))
     
;; (defun t/split (x sections &optional (axis 0))
;;  (let* ((_xs (t/u/split (data x) sections axis))
;;         (_masks (t/u/gen-mask 
;;                  _xs (lambda (ms) (numcl:concatenate ms :axis axis)))
;;    (loop for _x in _xs
;;          for _mask in _masks
;;          collect (let ((mask (t/new _mask)))
;;                    (t/prototype-op 
;;                     _x (t/op-bw t/bw/split x mask)))))

(defun t/split (x sections &optional (axis 0))
  (let* ((_xs (t/u/split (data x) sections axis))
         (_pads (mapcar #'numcl:zeros-like _xs))
         (pads (mapcar #'t/new _pads))
         (n (length _xs)))
    (loop for i from 0 to (- n 1)
          collect (let ((idx i)
                        (_x (nth i _xs)))
                    (t/prototype-op 
                     _x (t/op-bw t/bw/split x axis idx pads))))))

