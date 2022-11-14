;;;; tensor.lisp

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

(defun t/new (data)
  (let ((tmp (make-instance
              'tensor
              :data data
              :bwfn nil)))
    (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
    tmp))

(defun t/bw (out inps &optional (j (t/new 1)))
  (let ((gs (funcall (bwfn out) j)))
    (map 'list 
         (lambda (inp) 
           (reduce (lambda (prev vg) 
                     (if (eq inp (var-grad-tuple-v vg))
                         (t/+ prev (var-grad-tuple-g vg))
                         prev))
                   gs :initial-value (t/new 0))) inps)))

(defun t/+ (x y)
  (let ((tmp (make-instance
              'tensor
              :data (+ (data x) (data y))
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
                  :data (- (data x) (data y))
                  :bwfn nil)))
        (setf (bwfn tmp) 
              (lambda (j)
                (concatenate 'list
                             (list (make-var-grad-tuple :v tmp :g j))
                             (t/bw/- x y j))))
        tmp)
      (let ((tmp (make-instance
                  'tensor
                  :data (- (data x))
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
              :data (* (data x) (data y))
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
              :data (/ (data x) (data y))
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
              :data (expt (data x) (data y))
              :bwfn nil)))
    (setf (bwfn tmp) 
          (lambda (j)
            (concatenate 'list
                         (list (make-var-grad-tuple :v tmp :g j))
                         (t/bw/expt x y j))))
    tmp))

(defun t/log (x &optional (y (t/new (exp 1))))
  (let ((tmp (make-instance
              'tensor
              :data (log (data x) (data y))
              :bwfn nil)))
    (setf (bwfn tmp) 
          (lambda (j)
            (concatenate 'list
                         (list (make-var-grad-tuple :v tmp :g j))
                         (t/bw/log x y j))))
    tmp))

