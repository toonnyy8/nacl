;;; nerf.lisp

(ql:quickload "nacl")
(in-package nacl)

(ql:quickload "let-plus")
 
(ql:quickload "numpy-file-format")

(defvar poses (numcl:asarray (numpy-file-format:load-array "./test/nerf/data/poses.npy")))
(defvar images (numcl:asarray (numpy-file-format:load-array "./test/nerf/data/images.npy")))
(defvar focal (aref (numpy-file-format:load-array "./test/nerf/data/focal.npy")))

(defun posenc (x L &optional (step 1))
  (let* ((x-sin
           (loop for i from 0 to (- L 1)
                 collect 
                 (nacl:t/sin 
                  (nacl:t/* x (nacl:t/new `(,(expt 2 (* i step))))))))
         (x-cos
           (loop for i from 0 to (- L 1)
                 collect 
                 (nacl:t/cos 
                  (nacl:t/* x (nacl:t/new `(,(expt 2 (* i step))))))))
         (x-encs
           (concatenate 'list x-sin x-cos `(,x))))
    (nacl:t/cat x-encs -1)))
           
(defun get-rays (h w focal c2w)
  (let* ((i (t/new `(,(loop for i from 0 to (- w 1) 
                            collect (/ (- i (* w 0.5)) focal)))))
         (i (t/+ i (t/zeros `(,h 1))))
         (j (t/new (loop for j from 0 to (- h 1)
                         collect `(,(/ (- (* h 0.5) j) focal)))))
         (j (t/+ j (t/zeros `(1 ,w))))
         (dirs (t/stack `(,i ,j ,(t/full `(,h ,w) -1)) -1))
         (rays-d (t/matmul dirs (t/T (t/new (numcl:aref c2w '(t 3) '(t 3))))))
         (rays-o (t/+ (t/zeros `(,h ,w 3)) 
                      (t/reshape (t/new (numcl:aref c2w '(t 3) 3)) '(1 1 3)))))
   ;; (print (t/shape rays-d))
   ;; (print (t/shape rays-o))
   `(,(t/data rays-d) ,(t/data rays-o))))

(aref focal)

(numcl:uniform 0.0 1.0 '(10))
 

(numcl:aref poses 0) 
(numcl:aref poses 0 '(t 3) '(t 3))

(get-rays 32 32 focal (numcl:aref poses 0))

(defun render-rays (fwfn params rays-d rays-o near far 
                    n-samples &optional (use-rand nil))
  (let* ((z-vals (t/new (loop 
                          for i from 0 to (- n-samples 1)
                          collect (+ near (* (- far near) 
                                             (/ i (- n-samples 1)))))))
         (z-vals (t/reshape z-vals `(1 1 ,n-samples)))
         (h (nth 0 (t/shape rays-d)))
         (w (nth 1 (t/shape rays-d)))
         (z-vals (t/+ z-vals (if (not use-rand) 
                                 (t/zeros `(,h ,w ,n-samples)) 
                                 (t/rand `(,h ,w ,n-samples) 
                                         :high (/ (- far near) n-samples)))))
         (pts (t/+ (t/+ (t/reshape rays-d `(,h ,w 1 3))
                       (t/reshape rays-o `(,h ,w 1 3)))
                   (t/reshape z-vals `(,h ,w ,n-samples 1))))
         (pts (mapcar #'t/new (numcl:unstack (t/data pts) :axis 2)))
         (pts (mapcar (lambda (x) (funcall fwfn params x)) pts)))
    (t/shape (car pts))))


(defun nerf (in-dim out-dim hidden-dim n-layers 
             &optional (pe-L 6) (pe-step 1))
  (let* ((*in-dim (* in-dim (+ 1 (* 2 pe-L))))
         (blocks (loop for i from 0 to (- n-layers 1)
                       collect
                       (nn/seq
                        (nn/linear (+ *in-dim (if (= i 0) *in-dim hidden-dim))
                                   hidden-dim)
                        (nn/fn #'f/relu)
                        (nn/linear hidden-dim hidden-dim)
                        (nn/fn #'f/relu)
                        (nn/linear hidden-dim hidden-dim)
                        (nn/fn #'f/relu)
                        (nn/linear hidden-dim hidden-dim)
                        (nn/fn #'f/relu))))
         (out-layer (nn/linear hidden-dim out-dim)))
    (let ((initfn (lambda () 
                    (list (loop for b in blocks
                                collect (funcall (nth 0 b)))
                          (funcall (nth 0 out-layer)))))
          (fwfn (lambda (states x)
                  (let* ((*x (posenc x pe-L pe-step))
                         (inp *x))
                    (loop for b in blocks
                          for state in (car states)
                          do (setf inp (funcall (nth 1 b) state
                                                (t/cat `(,*x ,inp) -1))))
                    (funcall (nth 1 out-layer) (nth 1 states) inp)))))
      `(,initfn ,fwfn))))

(let* ((model (nerf 3 4 64 1))
       (optim (optim/adam 5e-4))
       (params (funcall (car model)))
       (fwfn (nth 1 model))
       (states (funcall (car optim) params))
       (stepfn (nth 1 optim)))
  (loop for i from 1 to 1
        do (let* ((sample (random 100))
                  (pose (numcl:aref poses sample))
                  ;; (image (numcl:aref images sample))
                  (w-start (random 24))
                  (h-start (random 24))
                  (rays-d-and-o (get-rays 32 32 focal pose))
                  (rays-d (numcl:aref (nth 0 rays-d-and-o)
                                      `(,h-start ,(+ h-start 8))
                                      `(,w-start ,(+ w-start 8))
                                      t))
                  (rays-d (t/new rays-d))
                  (rays-o (numcl:aref (nth 1 rays-d-and-o)
                                      `(,h-start ,(+ h-start 8))
                                      `(,w-start ,(+ w-start 8))
                                      t))
                  (rays-o (t/new rays-o)))
             (print (render-rays fwfn params rays-d rays-o 2 6 64)))))

(random 1)


;; (let* ((rays-d-and-o (get-rays 32 32 focal (numcl:aref poses 0)))
;;        (rays-d (nth 0 rays-d-and-o))
;;        (rays-o (nth 1 rays-d-and-o))
;;        (model (nerf 3 4 64 1))
;;        (optim (optim/adam 5e-4))
;;   (let* ((params (funcall (car model)))
;;          (fwfn (nth 1 model))
;;          (states (funcall (car optim) params))
;;          (stepfn (nth 1 optim))
;;     (render-rays fwfn params rays-d rays-o 2 6 4))
;; 
;; (let ((x (nacl:t/new '((1 2) (3 4)))))
;;   (nacl:t/shape (posenc x 6)))

