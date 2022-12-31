;;;; grad.lisp

(in-package #:nacl)

(defun t/bw/matmul (x y j)
  (concatenate 'list
               (funcall (bwfn x) (t/matmul j (t/T y)))
               (funcall (bwfn y) (t/matmul (t/T x) j))))

(defun t/bw/T (x j)
  (funcall (bwfn x) (t/T j)))

(defun t/bw/+ (x y j)
  (concatenate 'list
               (funcall (bwfn x) j)
               (funcall (bwfn y) j)))

(defun t/bw/- (x y j)
  (concatenate 'list
               (funcall (bwfn x) j)
               (funcall (bwfn y) (t/- j))))

(defun t/bw/neg (x j)
  (funcall (bwfn x) (t/- j)))

(defun t/bw/* (x y j)
  (concatenate 'list
               (funcall (bwfn x) (t/* j y))
               (funcall (bwfn y) (t/* j x))))

(defun t/bw/div (x y j)
  (concatenate 'list
               (funcall (bwfn x) (t/div j y))
               (funcall (bwfn y) (t/* (t/* j x) (t/- (t/expt y (t/full-like y -2)))))))

(defun t/bw/expt (x y j)
  (concatenate 'list
               (funcall (bwfn x) (t/* (t/* j y) (t/expt x (t/- y (t/ones-like y)))))
               (funcall (bwfn y) (t/* (t/* j (t/log x)) (t/expt x y)))))

(defun t/bw/log (x y j)
  (concatenate 'list
               (funcall (bwfn x) (t/div j (t/* (t/log y) x)))
               (funcall (bwfn y) (t/* j (t/div (t/- (t/log x)) (t/* y (t/* (t/log y) (t/log y))))))))

(defun t/bw/reshape (x j)
  (funcall (bwfn x) (t/reshape j (numcl:shape (data x)))))

(defun t/bw/sum (x keepdim-shape j)
  (funcall (bwfn x) (t/* (t/ones-like x) (t/reshape j keepdim-shape))))

(defun t/bw/mean (x keepdim-shape j)
  (let ((scalar (/ (reduce #'* (t/shape j) :initial-value 1)
                   (reduce #'* (t/shape x) :initial-value 1))))
    (funcall (bwfn x) (t/* (t/full-like x scalar) 
                           (t/reshape j keepdim-shape)))))

(defun t/bw/cos (x j)
  (funcall (bwfn x) (t/* j (t/- (t/sin x)))))

(defun t/bw/sin (x j)
  (funcall (bwfn x) (t/* j (t/cos x))))

(defun t/bw/stack (xs axis j)
  (let ((tmps nil))
    (loop for x in xs
          for _j in (t/unstack j axis)
          do (setf tmps (concatenate 'list tmps
                                     (funcall (bwfn x) _j))))
    tmps))

(defun t/bw/unstack (x axis idx pads j)
  (let* ((n (length pads))
         (pad-j (loop for i from 0 to (- n 1)
                      collect (if (= i idx) j (nth i pads)))))
    (funcall (bwfn x) (t/stack pad-j axis))))

(defun t/bw/cat (xs axis j)
  (let ((tmps nil)
        (sections (loop for x in xs 
                        collect (nth axis (numcl:shape (data x))))))
    (loop for x in xs
          for _j in (t/split j sections axis)
          do (setf tmps (concatenate 'list tmps
                                     (funcall (bwfn x) _j))))
    tmps))

;; (defun t/bw/split (x mask j)
;;   (funcall (bwfn x) (t/* j mask)))

(defun t/bw/split (x axis idx pads j)
  (let* ((n (length pads))
         (pad-j (loop for i from 0 to (- n 1)
                      collect (if (= i idx) j (nth i pads)))))
    (print (mapcar #'t/data pad-j))
    (funcall (bwfn x) (t/cat pad-j axis))))

