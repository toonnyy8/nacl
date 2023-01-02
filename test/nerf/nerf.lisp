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
   `(,(t/sg rays-d) ,(t/sg rays-o))))

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
         (pts (mapcar (lambda (x) (posenc (t/new x) 6))
                      (numcl:unstack (t/data pts) :axis 2))))
        pts))


(defun nerf (in-dim hidden-dim n-layers)
  ())

(let* ((rays-d-and-o (get-rays 32 32 focal (numcl:aref poses 0)))
       (rays-d (nth 0 rays-d-and-o))
       (rays-o (nth 1 rays-d-and-o))
       (model (nn/seq (nn/linear (+ 3 (* 3 2 6)) 128)
                      (nn/fn #'f/relu)
                      (nn/linear 128 128)
                      (nn/fn #'f/relu)
                      (nn/linear 128 128)
                      (nn/fn #'f/relu)
                      (nn/linear 16 1)
                      (nn/fn #'f/sigmoid)))
       (optim (optim/adam 0.005)))

  (render-rays nil nil rays-d rays-o 2 6 32))

(let ((x (nacl:t/new '((1 2) (3 4)))))
  (nacl:t/shape (posenc x 6)))

