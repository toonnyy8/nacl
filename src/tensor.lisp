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

;; (setq vg1 (make-var-grad-tuple :v (t/new 0) :g (t/new 2)))
;; 
;; (data (var-grad-tuple-v vg1))
;; (data (var-grad-tuple-g vg1))
;; 
;; (setq t1 (t/new 0))
;; (setq t2 (t/new 0))
;; (eq t1 t2)

(defun t/new (data)
  (let ((tmp (make-instance
              'tensor
              :data data
              :bwfn nil)))
    ;; (setf (bwfn tmp) (lambda (j) (setf (grad tmp) (+ (grad tmp) j)))) 
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

;; (defun t/- (x &optional y)
;;   (if y
;;       (make-instance
;;        'tensor
;;        :data (- (data x) (data y))
;;        :bwfn (lambda (j) (t/bw/- x y j)))
;;       (make-instance
;;        'tensor
;;        :data (- (data x))
;;        :bwfn (lambda (j) (t/bw/neg x j)))))


;; (defun t/- (x y)
;;   (make-instance
;;    'tensor
;;    :data (- (data x) (data y))
;;    :bwfn (lambda (j) (t/bw/- x y j))))

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

;; (defun t/* (x y)
;;   (make-instance
;;    'tensor
;;    :data (* (data x) (data y))
;;    :bwfn (lambda (j) (t/bw/* x y j))))

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

;; (defun t/div (x y)
;;   (make-instance
;;    'tensor
;;    :data (/ (data x) (data y))
;;    :bwfn (lambda (j) (t/bw/div x y j))))

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

;; (defun t/expt (x y)
;;   (make-instance
;;    'tensor
;;    :data (expt (data x) (data y))
;;    :bwfn (lambda (j) (t/bw/expt x y j))))

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

;; (let* ((a (new 2))
;;        (b (new 3))
;;        (c (+ a b)))
;;  (funcall (bwfn c) 1)
;;  (list (grad a) (grad b)))
;;  
;; (let* ((a (t/new 2))
;;        (b (t/new 3))
;;        (c (t/* a b)))
;;  (funcall (bwfn c) 1)
;;  (list (grad a) (grad b)))           
;; 
;; (data (t/new 1))
;; (bwfn t1)
;; 
;; (setf t1 (t/new 1))
;; 
;; (setf (data t1) 2)
;; (data t1)
;; (grad t1)
;; (funcall (bwfn t1) 1)

;; (defclass tensor ()
;;  ((shape
;;     :initarg :shape
;;     :accessor shape
;;   (data
;;     :initarg :data
;;     :accessor data
;;   (grad
;;     :initarg :grad
;;     :accessor grad
;;   (chain
;;     :initarg :chain
;;     :accessor chain)
;; 
;; (setf (fdefinition 't/data) #'data)
;; (setf (fdefinition 't/shape) #'shape)
;; (setf (fdefinition 't/grad) #'grad)
;; 
;; (defun t/ones (shape)
;;   (make-instance
;;     'tensor
;;     :data (numcl:ones shape)
;;     :shape shape
;;     :grad (numcl:zeros shape)
;;     :chain nil)
;; 
;; (defun t/zeros (shape)
;;   (make-instance
;;     'tensor
;;     :data (numcl:zeros shape)
;;     :shape shape
;;     :grad (numcl:zeros shape)
;;     :chain nil)
;; 
;; (defun t/randn (shape &key (mean 0.0) (var 1.0))
;;   (make-instance
;;     'tensor
;;     :data (numcl:normal
;;             (coerce mean 'double-float)
;;             (coerce var 'double-float)
;;             shape 'single-float
;;     :shape shape
;;     :grad (numcl:zeros shape)
;;     :chain nil)
