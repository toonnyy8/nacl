;;; optim.lisp

(in-package #:nacl)

(defun optim/sgd (&optional (lr 0.001))
  (let ((initfn (lambda (nn-states) nil))
        (stepfn (lambda (optim-states nn-states grads)
                  (let ((*nn-states (nn/u/flatten-states nn-states))
                        (*grads (nn/u/flatten-states grads)))
                    (list '()
                          (nn/u/shaping-states
                           nn-states
                           (loop for g in *grads
                                 for p in *nn-states
                                 collect
                                 (t/- p (t/* lr g)))))))))
    `(,initfn ,stepfn)))

(defun optim/adam (&optional (lr 0.001) 
                     (betas '(0.9 0.999))
                     (eps 1e-8))
  (let ((initfn (lambda (nn-states)
                  `(0 ;; iter num
                    ,(mapcar #'t/zeros-like (nn/u/flatten-states nn-states))
                    ,(mapcar #'t/zeros-like (nn/u/flatten-states nn-states)))))
        (stepfn (lambda (optim-states nn-states grads)
                  (let ((n (+ 1 (nth 0 optim-states)))
                        (ms_prev (nth 1 optim-states))
                        (vs_prev (nth 2 optim-states))
                        (*nn-states (nn/u/flatten-states nn-states))
                        (*grads (nn/u/flatten-states grads))
                        (beta1 (nth 0 betas))
                        (beta2 (nth 1 betas)))
                    (let* ((ms (loop for m_prev in ms_prev
                                     for g in *grads
                                     collect (t/+ (t/* beta1 m_prev)
                                                  (t/* (- 1 beta1) g))))
                           (vs (loop for v_prev in vs_prev
                                     for g in *grads
                                     collect (t/+ (t/* beta2 v_prev)
                                                  (t/* (- 1 beta2) (t/* g g)))))
                           (ms_hat (loop for m in ms
                                         collect 
                                         (t/div m (- 1 (expt beta1 n)))))
                           (vs_hat (loop for v in vs
                                         collect
                                         (t/gcp (t/div v (- 1 (expt beta2 n)))))))
                      (list `(,n ,ms ,vs)
                            (nn/u/shaping-states
                             nn-states
                             (loop for m_hat in ms_hat
                                   for v_hat in vs_hat
                                   for p in *nn-states
                                   collect
                                   (t/- p (t/div
                                           (t/* lr m_hat)
                                           (t/+ eps (t/expt v_hat 0.5))))))))))))
    `(,initfn ,stepfn)))
