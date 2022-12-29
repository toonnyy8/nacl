(ql:quickload "nacl")

(defun t/sum (x &optional (axes nil) &key (keepdim nil))
  (check-type x numcl:array)
  (let ((y (numcl:sum x :axes axes)))
    (if keepdim
        (let* ((org-shape (numcl:shape x)) 
               (new-shape (loop for dim in org-shape
                                for axis from 0 to (length org-shape)
                                collect (if (member axis axes) 1 dim))))
          (if (numcl:arrayp y)
              (numcl:reshape y new-shape) 
              (numcl:full new-shape y)))
        y)))

(t/sum (numcl:ones '(4 5 6)) '(2 0 1) :keepdim t)
(numcl:ones '(4 5 6))

(ql:quickload "iterate")
(loop for x in '(1 2 3)
      for i from 0 to (length '(1 2 3))
      collect (* x i))
(loop for x in '(1 2 3)
  do (print x))

(numcl)

(let* ((a 1)
       (a (* a 2))
       (a (* a 2))) 
  a)
(t/sum 2 '(1 2 3) :keepdim t)

(numcl:asarray (numcl:ones '(2)))
(numcl:ones '(2))
(numcl:asarray '(1 1))

(nacl:t/new '((1 2) (3 4)))

(numcl:< (numcl:* (numcl:ones '(2 3)) -1) (numcl:zeros '(2 3)))

(let* ((a (nacl:t/new '((1) (-2))))
       (b (nacl:t/new '((3 4))))
       (c (nacl:t/matmul a b))
       (d (nacl:f/relu c))
       (d (nacl:t/* d d)))
  (print (nacl:t/data d))
  ;;(nacl:t/bw d (list a b c)))
  (map 'list (lambda (x) (print (nacl:t/data x))) (nacl:t/bw d (list a b))))


(let* ((a (nacl:t/full '(1) 2.))
       (b (nacl:t/full '(1) 5.))
       (c (nacl:t/full '(1) 3.))
       (d (nacl:t/expt (nacl:t/+ a b) c)))
  ;;(nacl:t/bw d (list a b c)))
  (map 'list (lambda (x) (print (nacl:t/data x))) (nacl:t/bw d (list a b c))))

(let* ((a (nacl:t/full '(1) 10))
       (a³ (nacl:t/expt a (nacl:t/full '(1) 3)))
       (da (car (nacl:t/bw a³ (list a))))
       (d²a (car (nacl:t/bw da (list a))))
       (d³a (car (nacl:t/bw d²a (list a))))
       (d⁴a (car (nacl:t/bw d³a (list a)))))
  (list (nacl:t/data da)
        (nacl:t/data d²a)
        (nacl:t/data d³a)
        (nacl:t/data d⁴a)))

(let* ((a (nacl:t/full '(1) 10))
       (-a (nacl:t/- a))
       (da (car (nacl:t/bw a (list a))))
       (d-a (car (nacl:t/bw -a (list a)))))
  (list (nacl:t/data a)
        (nacl:t/data -a)
        (nacl:t/data da)
        (nacl:t/data d-a)))

