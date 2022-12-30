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

(defun t/shape (x)
  (check-type x tensor)
  (numcl:shape (t/data x)))

(defmacro make-tensor-prototype (body)
  `(make-instance
    'tensor
    :data ,body
    :bwfn nil))

(defstruct var-grad-tuple
  (v nil :type tensor)
  (g nil :type tensor))

(defmacro t/prototype-new (body)
  `(make-instance
    'tensor
    :data ,body
    :bwfn nil))

(defmacro t/prototype-op (fw bw)
  `(let ((tmp (make-instance
               'tensor
               :data ,fw
               :bwfn nil)))
     (setf (bwfn tmp) ,bw)
     tmp))

(defmacro t/init-bw () `(lambda (j) (list (make-var-grad-tuple :v tmp :g j))))

(defun t/new (array) (t/prototype-op (numcl:asarray array) (t/init-bw)))
;; (defun t/new (arr)
;;   (let ((tmp (t/prototype-new (numcl:asarray arr))))
;;     (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
;;     tmp))

(defun t/ones (shape) (t/prototype-op (numcl:ones shape) (t/init-bw)))
;; (defun t/ones (shape) (t/new (numcl:ones shape)))

(defun t/ones-like (array) 
  (t/prototype-op (numcl:ones-like (data array)) (t/init-bw)))
;; (defun t/ones-like (array)
;;   (let ((tmp (make-tensor-prototype (numcl:ones-like (data array)))))
;;     (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
;;     tmp))

(defun t/zeros (shape) (t/prototype-op (numcl:zeros shape) (t/init-bw)))
;; (defun t/zeros (shape)
;;   (let ((tmp (make-tensor-prototype (numcl:zeros shape))))
;;     (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
;;     tmp))

(defun t/zeros-like (array) 
  (t/prototype-op (numcl:zeros-like (data array)) (t/init-bw)))
;; (defun t/zeros-like (array)
;;   (let ((tmp (make-instance
;;               'tensor
;;               :data (numcl:zeros-like (data array))
;;               :bwfn nil)))
;;     (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
;;     tmp))

(defun t/randn (shape &key (mean 0.d0) (std 1.d0))
  (t/prototype-op (numcl:normal mean std shape) (t/init-bw)))
;; (defun t/randn (shape &key (mean 0.d0) (std 1.d0))
;;   (let ((tmp (make-instance
;;               'tensor
;;               :data (numcl:normal mean std shape)
;;               :bwfn nil)))
;;     (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
;;     tmp))

(defun t/randn-like (array &key (mean 0.d0) (std 1.d0))
  (t/prototype-op (numcl:normal mean std (numcl:shape (data array)))
                  (t/init-bw)))
;; (defun t/randn-like (array &key (mean 0.d0) (std 1.d0))
;;   (let ((tmp (make-instance
;;               'tensor
;;               :data (numcl:normal mean std (numcl:shape (data array)))
;;               :bwfn nil)))
;;     (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
;;     tmp))

(defun t/full (shape value)
  (t/prototype-op (numcl:full shape value) (t/init-bw)))
;; (defun t/full (shape value)
;;   (let ((tmp (make-instance
;;               'tensor
;;               :data (numcl:full shape value)
;;               :bwfn nil)))
;;     (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
;;     tmp))

(defun t/full-like (array value)
  (t/prototype-op (numcl:full (numcl:shape (data array)) value) (t/init-bw)))
;; (defun t/full-like (array value)
;;   (let ((tmp (make-instance
;;               'tensor
;;               :data (numcl:full-like (data array) value)
;;               :bwfn nil)))
;;     (setf (bwfn tmp) (lambda (j) (list (make-var-grad-tuple :v tmp :g j)))) 
;;     tmp))

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

;; (defmacro def-tensor-op (op-name body &rest args)
;;   (let ((t/op-name (intern (concatenate 'string "T/" (string op-name))))
;;         (t/bw/op-name (intern (concatenate 'string "T/BW/" (string op-name)))))
;;     `(defun ,t/op-name ,args
;;        (let ((tmp (make-instance
;;                    'tensor
;;                    :data ,body
;;                    :bwfn nil)))
;;         (setf (bwfn tmp)
;;               (lambda (j)
;;                 (concatenate 'list
;;                              (list (make-var-grad-tuple :v tmp :g j))
;;                              ,(concatenate 'list (list t/bw/op-name) args '(j)))))
;;         tmp))))

(defmacro tensor-op-body (op-name body &rest args)
  (let ((t/op-name (intern (concatenate 'string "T/" (string op-name))))
        (t/bw/op-name (intern (concatenate 'string "T/BW/" (string op-name)))))
    `(let ((tmp (make-tensor-prototype ,body)))
       (setf (bwfn tmp)
             (lambda (j)
               (concatenate 'list
                            (list (make-var-grad-tuple :v tmp :g j))
                            ,(concatenate 'list (list t/bw/op-name) args '(j)))))
       tmp)))

;; (defun t8/afn (a b) (- a b))
;; 
;; (defun callfn (name &rest args)
;;   (let ((t8/op (intern (concatenate 'string "T8/" (string name)))))
;;     (apply t8/op args)))
;; 
;; (callfn 'afn 1 2)
;; 
;; (concatenate 'list '(1 2) '())
;; 
;; (def-tensor-op matmul (numcl:matmul (data x) (data y)) x y)

(defmacro t/op-bw (bw-op-name &rest args)
  `(lambda (j) (concatenate 'list 
                            (list (make-var-grad-tuple :v tmp :g j))
                            ,(concatenate 'list (list bw-op-name) args '(j)))))

(defun t/u/matmul (x y)
  (let* ((x-shape (numcl:shape x))
         (y-shape (numcl:shape y))
         (x-dims (length x-shape))
         (y-dims (length y-shape))
         (max-dims (max x-dims y-dims))
         (x-shape (concatenate 'list 
                               (loop repeat (- max-dims x-dims) collect 1)
                               x-shape '(1)))
         (y-shape (concatenate 'list 
                               (loop repeat (+ 1 (- max-dims y-dims)) collect 1)
                               y-shape))
         (new-x (numcl:reshape x x-shape))
         (new-y (numcl:reshape y y-shape)))
    (numcl:sum (numcl:* new-x new-y) :axes (- max-dims 1))))

(defun t/matmul (x y)
  (t/prototype-op (t/u/matmul (data x) (data y))
                  (t/op-bw t/bw/matmul x y)))
;; (defun t/matmul (x y)
;;   (tensor-op-body matmul (numcl:matmul (data x) (data y)) x y))
;; (defun t/matmul (x y)
;;   (let ((tmp (make-instance
;;               'tensor
;;               :data (numcl:matmul (data x) (data y))
;;               :bwfn nil)))
;;     (setf (bwfn tmp) 
;;           (lambda (j)
;;             (concatenate 'list
;;                          (list (make-var-grad-tuple :v tmp :g j))
;;                          (t/bw/matmul x y j)))) 
;;     tmp))

(defun t/T (x)
  (t/prototype-op (numcl:transpose (data x))
                  (t/op-bw t/bw/T x)))
;; (defun t/T (x)
;;   (let ((tmp (t/new (numcl:transpose (data x)))))
;;     (setf (bwfn tmp) 
;;           (lambda (j)
;;             (concatenate 'list
;;                          (list (make-var-grad-tuple :v tmp :g j))
;;                          (t/bw/T x j)))) 
;;     tmp))

;; (defun t/T (x)
;;   (let ((tmp (make-instance
;;               'tensor
;;               :data (numcl:transpose (data x))
;;               :bwfn nil)))
;;     (setf (bwfn tmp) 
;;           (lambda (j)
;;             (concatenate 'list
;;                          (list (make-var-grad-tuple :v tmp :g j))
;;                          (t/bw/T x j)))) 
;;     tmp))

(defun t/+ (x y)
  (t/prototype-op (numcl:+ (data x) (data y))
                  (t/op-bw t/bw/+ x y)))
;; (defun t/+ (x y)
;;   (tensor-op-body + (numcl:+ (data x) (data y)) x y))
;; (def-tensor-op + (numcl:+ (data x) (data y)) x y)

;; (defun t/+ (x y)
;;   (let ((tmp (make-instance
;;               'tensor
;;               :data (numcl:+ (data x) (data y))
;;               :bwfn nil)))
;;     (setf (bwfn tmp) 
;;           (lambda (j)
;;             (concatenate 'list
;;                          (list (make-var-grad-tuple :v tmp :g j))
;;                          (t/bw/+ x y j)))) 
;;     tmp))

(defun t/- (x &optional y)
  (if y
      (t/prototype-op (numcl:- (data x) (data y))
                  (t/op-bw t/bw/- x y))
      (t/prototype-op (numcl:- (data x))
                  (t/op-bw t/bw/neg x))))
;; (defun t/- (x &optional y)
;;   (if y
;;       (let ((tmp (make-instance
;;                   'tensor
;;                   :data (numcl:- (data x) (data y))
;;                   :bwfn nil)))
;;         (setf (bwfn tmp) 
;;               (lambda (j)
;;                 (concatenate 'list
;;                              (list (make-var-grad-tuple :v tmp :g j))
;;                              (t/bw/- x y j))))
;;         tmp)
;;       (let ((tmp (make-instance
;;                   'tensor
;;                   :data (numcl:- (data x))
;;                   :bwfn nil)))
;;         (setf (bwfn tmp) 
;;               (lambda (j)
;;                 (concatenate 'list
;;                              (list (make-var-grad-tuple :v tmp :g j))
;;                              (t/bw/neg x j))))
;;         tmp)))

(defun t/* (x y)
  (t/prototype-op (numcl:* (data x) (data y))
                  (t/op-bw t/bw/* x y)))
;; (defun t/* (x y)
;;   (let ((tmp (make-instance
;;               'tensor
;;               :data (numcl:* (data x) (data y))
;;               :bwfn nil)))
;;     (setf (bwfn tmp) 
;;           (lambda (j)
;;             (concatenate 'list
;;                          (list (make-var-grad-tuple :v tmp :g j))
;;                          (t/bw/* x y j))))
;;     tmp))

(defun t/div (x y)
  (t/prototype-op (numcl:/ (data x) (data y))
                  (t/op-bw t/bw/div x y)))
;; (defun t/div (x y)
;;   (let ((tmp (make-instance
;;               'tensor
;;               :data (numcl:/ (data x) (data y))
;;               :bwfn nil)))
;;     (setf (bwfn tmp) 
;;           (lambda (j)
;;             (concatenate 'list
;;                          (list (make-var-grad-tuple :v tmp :g j))
;;                          (t/bw/div x y j))))
;;     tmp))

(defun t/expt (x y)
  (t/prototype-op (numcl:expt (data x) (data y))
                  (t/op-bw t/bw/expt x y)))
;; (defun t/expt (x y)
;;   (let ((tmp (make-instance
;;               'tensor
;;               :data (numcl:expt (data x) (data y))
;;               :bwfn nil)))
;;     (setf (bwfn tmp) 
;;           (lambda (j)
;;             (concatenate 'list
;;                          (list (make-var-grad-tuple :v tmp :g j))
;;                          (t/bw/expt x y j))))
;;     tmp))

(defun t/u/log (x &optional y)
  (if (not y) (setq y (numcl:exp (numcl:ones-like x))))
  (numcl:/ (numcl:log x) (numcl:log y)))

(defun t/log (x &optional y)
  (if (not y) (setq y (t/ones-like x)))
  (t/prototype-op (t/u/log (data x) (data y))
                  (t/op-bw t/bw/log x y)))
;; (defun t/log (x &optional y)
;;   (if (not y) (setq y (t/ones-like x)))
;;   (let ((tmp (make-instance
;;               'tensor
;;               :data (numcl/log (data x) (data y))
;;               :bwfn nil)))
;;     (setf (bwfn tmp) 
;;           (lambda (j)
;;             (concatenate 'list
;;                          (list (make-var-grad-tuple :v tmp :g j))
;;                          (t/bw/log x y j))))
;;     tmp))

(defun t/reshape (x shape)
  (t/prototype-op (numcl:reshape (data x) shape)
                  (t/op-bw t/bw/reshape x)))

(defun t/u/reduce-shape (x axes) 
  (let ((org-shape (numcl:shape x)))
    (loop for dim in org-shape
          for axis from 0 to (length org-shape)
          collect (if (member axis axes) 1 dim))))

(defun t/u/sum (x &optional (axes nil) &key (keepdim nil))
  (check-type x numcl:array)
  (let ((y (numcl:sum x :axes axes)))
    (if keepdim
        (let ((new-shape (t/u/reduce-shape x axes)))
          (if (numcl:arrayp y)
              (numcl:reshape y new-shape) 
              (numcl:full new-shape y)))
        y)))
(defun t/sum (x &optional (axes nil) &key (keepdim nil))
  (t/prototype-op (t/u/sum (data x) axes :keepdim keepdim)
                  (t/op-bw t/bw/sum x (t/u/reduce-shape (data x) axes))))


(defun t/u/mean (x &optional (axes nil) &key (keepdim nil))
  (check-type x numcl:array)
  (let ((y (numcl:mean x :axes axes)))
    (if keepdim
        (let ((new-shape (t/u/reduce-shape x axes)))
          (if (numcl:arrayp y)
              (numcl:reshape y new-shape) 
              (numcl:full new-shape y)))
        y)))
(defun t/mean (x &optional (axes nil) &key (keepdim nil))
  (t/prototype-op (t/u/mean (data x) axes :keepdim keepdim)
                  (t/op-bw t/bw/mean x (t/u/reduce-shape (data x) axes))))


;; (defmacro def-tensor-op (op-name bw-op-name body &rest args)
;;   (list 'defun op-name args
;;     (list 'let (list (list 'tmp (list 'make-instance
;;                                                     ''tensor
;;                                                     :data body
;;                                                     :bwfn nil)))
;;       (list 'setf (list 'bwfn 'tmp) 
;;             (list 'lambda '(j)
;;               (list 'concatenate ''list
;;                            (list 'list '(make-var-grad-tuple :v tmp :g j))
;;                            (concatenate 'list (list bw-op-name) args '(j))))) 
;;       'tmp)))

;; (defmacro def-tensor-op (op-name body &rest args)
;;   (let ((t/op-name (intern (concatenate 'string "T/" (string op-name))))
;;         (t/bw/op-name (intern (concatenate 'string "T/BW" (string op-name)))))
;;     (list 'defun t/op-name args
;;       (list 'let (list (list 'tmp (list 'make-instance
;;                                                       ''tensor
;;                                                       :data body
;;                                                       :bwfn nil)))
;;         (list 'setf (list 'bwfn 'tmp) 
;;               (list 'lambda '(j)
;;                 (list 'concatenate ''list
;;                              (list 'list '(make-var-grad-tuple :v tmp :g j))
;;                              (concatenate 'list (list t/bw/op-name) args '(j))))) 
;;         'tmp))))



;; (defmacro def-tensor-op (op-name body &rest args)
;;   (let ((t8/op-name (intern (concatenate 'string "T/" (string op-name))))
;;         (t8/bw/op-name (intern (concatenate 'string "T/BW" (string op-name)))))
;;     (list 'defun t8/op-name args
;;       (list 'let (list (list 'tmp (list 'make-instance
;;                                         ''tensor
;;                                         :data body
;;                                         :bwfn nil)))))))

;; (def-tensor-op ttt (+ x y) x y)
;; (t/ttt 1 2)
;; (defmacro catname (name) (intern (concatenate 'string "T/" (string name))))
;; (let (((catname ++--) 1)))
;; (intern (concatenate 'string "T/" (string '++--)))
;; (t/ttt 1 2)
;; (macroexpand (def-tensor-op t/ttt t/bw/ttt (numcl:+ x y) x y))
;; (concatenate 'list (list 1 2 (+ 3 1)) '(6))
;; (list 'lambda '(l))
;; 
;; (defun afn (let ((a 1))))

