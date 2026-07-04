;;; second-mind.el --- Query the local Second Mind /ask endpoint -*- lexical-binding: t; -*-

;; Author: Second Mind
;; Version: 0.1.0
;; Package-Requires: ((emacs "27.1"))
;; Keywords: convenience, outlines, hypermedia

;;; Commentary:

;; A thin Emacs client for the local `inbox' second-brain HTTP endpoint
;; (`POST /ask', KB-only RAG over the org corpus).  `M-x second-mind-ask'
;; prompts for a question, queries the endpoint, and shows the answer with
;; clickable `[[id:...]]' org-roam citations in an org buffer.
;;
;; The endpoint binds loopback by default (see `admin.ask_bind' in the inbox
;; config), so this only works from the machine running the daemon.
;;
;; Install: put this file on `load-path' and (require 'second-mind), or
;; symlink it into your Emacs config's Lisp directory.

;;; Code:

(require 'json)
(require 'url)
(require 'org)
(require 'subr-x)

(defgroup second-mind nil
  "Query the local Second Mind (inbox) second-brain endpoint."
  :group 'external
  :prefix "second-mind-")

(defcustom second-mind-ask-url "http://127.0.0.1:9091/ask"
  "URL of the local /ask endpoint.
Loopback by default: the endpoint serves the full private KB with no
authentication, so it must not be exposed off the local machine."
  :type 'string)

(defcustom second-mind-top-k 6
  "Default number of KB chunks to retrieve per query."
  :type 'integer)

(defcustom second-mind-timeout 60
  "Seconds to wait for the /ask endpoint before giving up."
  :type 'integer)

(defcustom second-mind-buffer-name "*Second Mind*"
  "Name of the buffer used to display answers."
  :type 'string)

(defun second-mind--request (question top-k)
  "POST QUESTION (retrieving TOP-K chunks) and return the parsed response alist.
Signals a `user-error' on transport or server failure."
  (let* ((url-request-method "POST")
         (url-request-extra-headers '(("Content-Type" . "application/json")))
         (url-request-data
          (encode-coding-string
           (json-encode (list (cons "question" question)
                              (cons "top_k" top-k)))
           'utf-8))
         (buffer (url-retrieve-synchronously
                  second-mind-ask-url t t second-mind-timeout)))
    (unless (buffer-live-p buffer)
      (user-error "Second Mind: no response from %s (is the daemon running?)"
                  second-mind-ask-url))
    (unwind-protect
        (with-current-buffer buffer
          (goto-char (point-min))
          (let ((status-line (buffer-substring-no-properties
                              (point) (line-end-position))))
            (unless (re-search-forward "\r?\n\r?\n" nil t)
              (user-error "Second Mind: malformed HTTP response"))
            (let ((body (string-trim
                         (buffer-substring-no-properties (point) (point-max)))))
              (unless (string-match-p " 200 " status-line)
                (user-error "Second Mind: %s — %s" status-line body))
              (json-parse-string body :object-type 'alist :array-type 'list))))
      (kill-buffer buffer))))

;;;###autoload
(defun second-mind-ask (question &optional top-k)
  "Ask the local Second Mind QUESTION and show the answer with citations.
With a prefix argument, also prompt for TOP-K (chunks to retrieve)."
  (interactive
   (list (read-string "Second Mind — ask: ")
         (when current-prefix-arg
           (read-number "top_k: " second-mind-top-k))))
  (when (string-empty-p (string-trim question))
    (user-error "Second Mind: question must not be empty"))
  (let* ((k (or top-k second-mind-top-k))
         (response (second-mind--request question k))
         (org (alist-get 'org response))
         (buffer (get-buffer-create second-mind-buffer-name)))
    (with-current-buffer buffer
      (let ((inhibit-read-only t))
        (erase-buffer)
        (org-mode)
        (insert "#+title: Second Mind\n\n")
        (insert "* Q: " question "\n\n")
        (insert (if (and org (stringp org)) org "(no answer)") "\n"))
      (goto-char (point-min)))
    (pop-to-buffer buffer)))

(provide 'second-mind)
;;; second-mind.el ends here
