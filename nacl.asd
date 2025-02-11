 ;;; nacl.asd

(asdf:defsystem #:nacl
  :description "Describe NaCL here"
  :author "Your Name <your.name@example.com>"
  :license  "Specify license here"
  :version "0.0.1"
  :serial t
  :depends-on (#:numcl
               #:float-features)
  :components ((:module "src"
                :serial t
                :components ((:file "package")
                             (:file "nacl")
                             (:file "tensor")
                             (:file "grad")
                             (:file "fn")
                             (:file "nn")
                             (:file "optim")))))
