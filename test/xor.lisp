(ql:quickload "nacl")

(in-package nacl)

(let* ((x (t/new '((0 0)
                   (1 1)
                   (1 0)
                   (0 1))))
       (y (t/new '((0) (0) (1) (1))))
       (model (nn/seq (nn/linear 2 16)
                      (nn/fn #'f/relu)
                      (nn/linear 16 16)
                      (nn/fn #'f/relu)
                      (nn/linear 16 16)
                      (nn/fn #'f/relu)
                      (nn/linear 16 1)
                      (nn/fn #'f/sigmoid)))
       (optim (optim/adam 0.005)))
  (let* ((params (funcall (nth 0 model)))
         (fwfn (nth 1 model))
         (states (funcall (nth 0 optim) params))
         (stepfn (nth 1 optim)))
    (print (t/shape (funcall fwfn params x)))
    (print params)
    (loop for i from 1 to 200
          do (let* ((pred (funcall fwfn params x))
                    (loss (t/mean (f/bce pred y)))
                    (grads (t/bw loss params))
                    (states-and-params (funcall stepfn states params grads)))
               (setf states (t/sg (nth 0 states-and-params)))
               (setf params (t/sg (nth 1 states-and-params)))
               (if (= 0 (mod i 10)) (print (t/data loss)))))
    (print (t/data (funcall fwfn params x)))))
