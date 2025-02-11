(ql:quickload "nacl")

(in-package nacl)

(defun tgtfn (x) (t/expt x 0.5))

(defvar fwfn)
(defvar params)

(setf (values fwfn params)
  (let* ((x (t/new '((0) (1) (4) (9) (16) (25) (36) (49) (64) (81) (100))))
         (y (t/sg (tgtfn x)))
         (model (nn/seq (nn/linear 1 16)
                        (nn/fn #'f/relu)
                        (nn/linear 16 16)
                        (nn/fn #'f/relu)
                        (nn/linear 16 16)
                        (nn/fn #'f/relu)
                        (nn/linear 16 16)
                        (nn/fn #'f/relu)
                        (nn/linear 16 16)
                        (nn/fn #'f/relu)
                        (nn/linear 16 16)
                        (nn/fn #'f/relu)
                        (nn/linear 16 1)))
         (optim (optim/adam)))
    (let* ((params (funcall (nth 0 model)))
           (fwfn (nth 1 model))
           (states (funcall (nth 0 optim) params))
           (stepfn (nth 1 optim)))
      (print (t/data x))
      (print (t/data y))
      (print params)
      (loop for i from 1 to 200
            do (let* ((pred (funcall fwfn params x))
                      (loss (t/mean (t/expt (t/- pred y) 2)))
                      (grads (t/bw loss params))
                      (states-and-params (funcall stepfn states params grads)))
                 (setf states (t/sg (nth 0 states-and-params)))
                 (setf params (t/sg (nth 1 states-and-params)))
                 (if (= 0 (mod i 10)) (print (t/data loss)))))
      (values fwfn params))))

(let* ((test-x (t/new '((0.5) (6.7) (144) (225))))
       (test-y (t/sg (tgtfn test-x)))
       (pred (funcall fwfn params test-x))
       (loss (t/mean (t/expt (t/- pred test-y) 2))))
  (print "test loss")
  (print (t/data loss))
  (print (t/data (t/reshape pred '(-1))))
  (print (t/data (t/reshape test-y '(-1)))))
