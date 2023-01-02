;;;; grad.lisp

(in-package #:nacl)

;; (defun t/bw/matmul (x y j)
;;   (concatenate 'list
;;                (funcall (bwfn x) (t/matmul j (t/T y)))
;;                (funcall (bwfn y) (t/matmul (t/T x) j)))

(defun t/bw/T (x j)
  (funcall (bwfn x) (t/T j)))

(defun t/u/broadcast-axes (src tgt)
  (let* ((src-shape (t/shape src))
         (n (length src-shape))
         (tgt-shape (t/shape tgt))
         (m (length tgt-shape)))
    (if (and (= 1 n) (= (car src-shape) 1))
        (loop for i from 0 to (- m 1)
              collect i)
        (let ((axes nil))
          (loop for i from 0 to (- m 1)
                do (if (not (= (nth i src-shape) (nth i tgt-shape)))
                       (setf axes (concatenate 'list axes `(,i)))))
          axes))))

;; (t/u/broadcast-axes (t/ones '(2 3 4)) (t/ones '(2 3 4)))

(defun t/u/inv-broadcast-grad (x jx)
  (let ((broadcast-axes (t/u/broadcast-axes x jx)))
    (if broadcast-axes 
        (t/reshape (t/sum jx broadcast-axes) (t/shape x))
        jx)))

(defun t/bw/+ (x y j)
  (let* ((jx j)
         (jy j))
    (concatenate 'list
                 (funcall (bwfn x) (t/u/inv-broadcast-grad x jx))
                 (funcall (bwfn y) (t/u/inv-broadcast-grad y jy)))))

(defun t/bw/- (x y j)
  (let* ((jx j) 
         (jy (t/- j)))
    (concatenate 'list
                 (funcall (bwfn x) (t/u/inv-broadcast-grad x jx))
                 (funcall (bwfn y) (t/u/inv-broadcast-grad y jy)))))

(defun t/bw/neg (x j)
  (funcall (bwfn x) (t/- j)))

(defun t/bw/* (x y j)
  (let* ((jx (t/* j y))
         (jy (t/* j x)))
    (concatenate 'list
                 (funcall (bwfn x) (t/u/inv-broadcast-grad x jx))
                 (funcall (bwfn y) (t/u/inv-broadcast-grad y jy)))))

(defun t/bw/div (x y j)
  (let* ((jx (t/div j y))
         (jy (t/* (t/* j x) (t/- (t/expt y (t/full-like y -2))))))
    (concatenate 'list
                (funcall (bwfn x) (t/u/inv-broadcast-grad x jx))
                (funcall (bwfn y) (t/u/inv-broadcast-grad y jy)))))

(defun t/bw/expt (x y j)
  (let* ((jx (t/* (t/* j y) (t/expt x (t/- y (t/ones-like y)))))
         (jy (t/* (t/* j (t/log x)) (t/expt x y))))
    (concatenate 'list
                (funcall (bwfn x) (t/u/inv-broadcast-grad x jx))
                (funcall (bwfn y) (t/u/inv-broadcast-grad y jy)))))

(defun t/bw/log (x y j)
  (let* ((jx (t/div j (t/* (t/log y) x)))
         (jy (t/* j (t/div (t/- (t/log x)) 
                           (t/* y (t/* (t/log y) (t/log y)))))))
    (concatenate 'list
                (funcall (bwfn x) (t/u/inv-broadcast-grad x jx))
                (funcall (bwfn y) (t/u/inv-broadcast-grad y jy)))))

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
    ;; (print (mapcar #'t/data pad-j))
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
    (funcall (bwfn x) (t/cat pad-j axis))))

