;;;; grad.lisp

(in-package #:nacl)

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
               (funcall (bwfn y) (t/* (t/* j x) (t/- (t/expt y (t/new -2)))))))

(defun t/bw/expt (x y j)
  (concatenate 'list
               (funcall (bwfn x) (t/* (t/* j y) (t/expt x (t/- y (t/new 1)))))
               (funcall (bwfn y) (t/* (t/* j (t/log x)) (t/expt x y)))))

(defun t/bw/log (x y j)
  (concatenate 'list
               (funcall (bwfn x) (t/div j (t/* (t/log y) x)))
               (funcall (bwfn y) (t/* j (t/div (t/- (t/log x)) (t/* y (t/* (t/log y) (t/log y))))))))
