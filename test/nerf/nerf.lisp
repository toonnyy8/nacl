;;; nerf.lisp

(ql:quickload "nacl")

(ql:quickload "let-plus")
(in-package let-plus)
 
(ql:quickload "numpy-file-format")

(let+ (((&values a b) (nacl:nn/linear 2 32)))
  (funcall a))

(let+ (((&values a &ign b) (values 1 2 3)))
  (list a b))

(nacl:nn/linear 2 32)

(numcl:shape (numcl:asarray (numpy-file-format:load-array "./test/nerf/data/poses.npy")))
(numcl:shape (numcl:asarray (numpy-file-format:load-array "./test/nerf/data/images.npy")))
(numpy-file-format:load-array "./test/nerf/data/focal.npy")

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
           

(let ((x (nacl:t/new '((1 2) (3 4)))))
  (nacl:t/shape (posenc x 6)))
