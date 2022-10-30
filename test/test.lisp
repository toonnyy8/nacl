(pushnew (uiop:getcwd) asdf:*central-registry*)
(pushnew #P"/root/git/nacl/" asdf:*central-registry*)
(ql:quickload "nacl")

(defclass tensor ()
 ((shape
   :initarg :shape
   :accessor shape)
  (value
   :initarg :value
   :accessor value)
  (grad
   :initarg :grad
   :accessor grad)))

(defun t/ones (shape)
 (make-instance
  'tensor
  :value (numcl:ones shape)
  :shape shape
  :grad (numcl:zeros shape)))

(value (t/ones '(3 3)))
(shape (t/ones '(3 3)))
(grad (t/ones '(3 3)))

(nacl:add 1 2)

(nacl:sub 1 2)
(nacl:t/ones)
(nacl:t/zeros)
 
