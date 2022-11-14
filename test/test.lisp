(ql:quickload "nacl")

(let* ((a (nacl:t/new 2))
       (b (nacl:t/new 5))
       (c (nacl:t/new 3))
       (d (nacl:t/expt (nacl:t/+ a b) c)))
  (map 'list (lambda (x) (print (nacl:t/data x))) (nacl:t/bw d (list a b c))))

(let* ((a (nacl:t/new 10))
       (a³ (nacl:t/expt a (nacl:t/new 3)))
       (da (car (nacl:t/bw a³ (list a))))
       (d²a (car (nacl:t/bw da (list a))))
       (d³a (car (nacl:t/bw d²a (list a))))
       (d⁴a (car (nacl:t/bw d³a (list a)))))
  (list (nacl:t/data da)
        (nacl:t/data d²a)
        (nacl:t/data d³a)
        (nacl:t/data d⁴a)))

