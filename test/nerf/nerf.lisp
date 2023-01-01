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
 
