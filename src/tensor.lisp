;;;; tensor.lisp

(ql:quickload :numcl)

(in-package nacl)

(let ((tid 0))
  (defun t/id () (setq tid (1+ tid)) (1- tid)))

(defclass tensor ()
  ((data
    :initarg :data
    :accessor data)
   (tid 
    :initform (t/id)
    :accessor tid)
   (bwfn
    :initarg :bwfn
    :accessor bwfn)))

(setf (fdefinition 't/data) #'data)
(setf (fdefinition 't/tid) #'tid)
(setf (fdefinition 't/bwfn) #'bwfn)

(defstruct var-grad-tuple
  (v nil :type tensor)
  (g nil :type tensor))

(defun t/ones (shape)
  (let ((tmp (make-instance
              'tensor
              :data (numcl:ones shape)
              :bwfn nil)))
    (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
    tmp))

(defun t/ones-like (array)
  (let ((tmp (make-instance
              'tensor
              :data (numcl:ones-like (data array))
              :bwfn nil)))
    (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
    tmp))

(defun t/zeros (shape)
  (let ((tmp (make-instance
              'tensor
              :data (numcl:zeros shape)
              :bwfn nil)))
    (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
    tmp))

(defun t/zeros-like (array)
  (let ((tmp (make-instance
              'tensor
              :data (numcl:zeros-like (data array))
              :bwfn nil)))
    (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
    tmp))

(defun t/randn (shape &key (mean 0.d0) (std 1.d0))
  (let ((tmp (make-instance
              'tensor
              :data (numcl:normal mean std shape)
              :bwfn nil)))
    (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
    tmp))

(defun t/randn-like (array &key (mean 0.d0) (std 1.d0))
  (let ((tmp (make-instance
              'tensor
              :data (numcl:normal mean std (numcl:shape (data array)))
              :bwfn nil)))
    (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
    tmp))

(defun t/full (shape value)
  (let ((tmp (make-instance
              'tensor
              :data (numcl:full shape value)
              :bwfn nil)))
    (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
    tmp))

(defun t/full-like (array value)
  (let ((tmp (make-instance
              'tensor
              :data (numcl:full-like (data array) value)
              :bwfn nil)))
    (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
    tmp))

(defun t/bw (out inps &optional j)
  (if (not j) (setq j (t/ones-like out)))
  (let ((gs (funcall (bwfn out) j)))
    (map 'list 
         (lambda (inp) 
           (reduce (lambda (prev vg) 
                     (if (eq inp (var-grad-tuple-v vg))
                         (t/+ prev (var-grad-tuple-g vg))
                         prev))
                   gs :initial-value (t/zeros-like inp))) inps)))

;; (defun t/bw (out inps &optional j)
;;   (if (not j) (setq j (numcl:ones-like out)))
;;   (let ((gs (funcall (bwfn out) j)))
;;     (map 'list 
;;          (lambda (inp) 
;;            (reduce (lambda (prev vg) 
;;                      (if (eq inp (var-grad-tuple-v vg))
;;                          (t/+ prev (var-grad-tuple-g vg))
;;                          prev
;;                    gs :initial-value (t/new 0))) inps)

(defun t/matmul (x y)
  (let ((tmp (make-instance
              'tensor
              :data (numcl:matmul (data x) (data y))
              :bwfn nil)))
    (setf (bwfn tmp) 
          (lambda (j)
            (concatenate 'list
                         (list (make-var-grad-tuple :v tmp :g j))
                         (t/bw/matmul x y j)))) 
    tmp))


(defun t/T (x)
  (let ((tmp (make-instance
              'tensor
              :data (numcl:transpose (data x))
              :bwfn nil)))
    (setf (bwfn tmp) 
          (lambda (j)
            (concatenate 'list
                         (list (make-var-grad-tuple :v tmp :g j))
                         (t/bw/T x j)))) 
    tmp))


(defun t/msquare (x)
  (let ((tmp (make-instance
              'tensor
              :data (numcl:square (data x))
              :bwfn nil)))
    (setf (bwfn tmp) 
          (lambda (j)
            (concatenate 'list
                         (list (make-var-grad-tuple :v tmp :g j))
                         (t/bw/msquare x j)))) 
    tmp))

(defun t/+ (x y)
  (let ((tmp (make-instance
              'tensor
              :data (numcl:+ (data x) (data y))
              :bwfn nil)))
    (setf (bwfn tmp) 
          (lambda (j)
            (concatenate 'list
                         (list (make-var-grad-tuple :v tmp :g j))
                         (t/bw/+ x y j)))) 
    tmp))

(defun t/- (x &optional y)
  (if y
      (let ((tmp (make-instance
                  'tensor
                  :data (numcl:- (data x) (data y))
                  :bwfn nil)))
        (setf (bwfn tmp) 
              (lambda (j)
                (concatenate 'list
                             (list (make-var-grad-tuple :v tmp :g j))
                             (t/bw/- x y j))))
        tmp)
      (let ((tmp (make-instance
                  'tensor
                  :data (numcl:- (data x))
                  :bwfn nil)))
        (setf (bwfn tmp) 
              (lambda (j)
                (concatenate 'list
                             (list (make-var-grad-tuple :v tmp :g j))
                             (t/bw/neg x j))))
        tmp)))

(defun t/* (x y)
  (let ((tmp (make-instance
              'tensor
              :data (numcl:* (data x) (data y))
              :bwfn nil)))
    (setf (bwfn tmp) 
          (lambda (j)
            (concatenate 'list
                         (list (make-var-grad-tuple :v tmp :g j))
                         (t/bw/* x y j))))
    tmp))

(defun t/div (x y)
  (let ((tmp (make-instance
              'tensor
              :data (numcl:/ (data x) (data y))
              :bwfn nil)))
    (setf (bwfn tmp) 
          (lambda (j)
            (concatenate 'list
                         (list (make-var-grad-tuple :v tmp :g j))
                         (t/bw/div x y j))))
    tmp))

(defun t/expt (x y)
  (let ((tmp (make-instance
              'tensor
              :data (numcl:expt (data x) (data y))
              :bwfn nil)))
    (setf (bwfn tmp) 
          (lambda (j)
            (concatenate 'list
                         (list (make-var-grad-tuple :v tmp :g j))
                         (t/bw/expt x y j))))
    tmp))

(defun numcl/log (x &optional y)
  (if (not y) (setq y (numcl:exp (numcl:ones-like x))))
  (numcl:/ (numcl:log x) (numcl:log y)))

(defun t/log (x &optional y)
  (if (not y) (setq y (t/ones-like x)))
  (let ((tmp (make-instance
              'tensor
              :data (numcl/log (data x) (data y))
              :bwfn nil)))
    (setf (bwfn tmp) 
          (lambda (j)
            (concatenate 'list
                         (list (make-var-grad-tuple :v tmp :g j))
                         (t/bw/log x y j))))
    tmp))

