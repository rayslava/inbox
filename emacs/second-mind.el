;;; second-mind.el --- Query the local Second Mind /ask endpoint -*- lexical-binding: t; -*-

;; Author: Second Mind
;; Version: 0.2.0
;; Package-Requires: ((emacs "27.1"))
;; Keywords: convenience, outlines, hypermedia

;;; Commentary:

;; A thin Emacs client for the local `inbox' second-brain HTTP endpoint
;; (`POST /ask', per-kind RAG over the org corpus).  It offers three query
;; surfaces over the same knowledge base:
;;
;; - `M-x second-mind-ask'  — semantic RAG: prompts for a question, queries
;;   the endpoint, and shows the answer with clickable `[[id:...]]' org-roam
;;   citations.  With a prefix argument, also prompts for `top_k' and the
;;   retrieval mode (memory / kb / hybrid; see `second-mind-default-mode').
;; - `M-x second-mind-ql'  — structured search: runs an `org-ql' query over the
;;   same org directory for exact tag/todo/property matches (optional dep).
;; - `M-x second-mind-insert-answer'  — inserts a KB-cited answer at point, e.g.
;;   to ground a `gptel' conversation before chatting.
;;
;; The endpoint binds loopback by default (see `admin.ask_bind' in the inbox
;; config), so this only works from the machine running the daemon.
;;
;; Install: put this file on `load-path' and (require 'second-mind), or
;; symlink it into your Emacs config's Lisp directory.  `org-ql' is required
;; only for `second-mind-ql'; `gptel' is not required at all —
;; `second-mind-insert-answer' just inserts text.

;;; Code:

(require 'json)
(require 'url)
(require 'org)
(require 'subr-x)

(declare-function org-ql-search "ext:org-ql-search")

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

(defcustom second-mind-default-mode "kb"
  "Default retrieval scope for a query.
One of \"memory\" (behavioral memory only), \"kb\" (KB chunks, the RAG
default), or \"hybrid\" (a quota'd blend tuned for vague queries)."
  :type '(choice (const "memory") (const "kb") (const "hybrid")))

(defcustom second-mind-timeout 60
  "Seconds to wait for the /ask endpoint before giving up."
  :type 'integer)

(defcustom second-mind-buffer-name "*Second Mind*"
  "Name of the buffer used to display answers."
  :type 'string)

(defcustom second-mind-org-directory nil
  "Directory of org files the KB indexes, for `second-mind-ql'.
When nil, falls back to `org-roam-directory' then `org-directory'."
  :type '(choice (const :tag "Auto (org-roam/org-directory)" nil) directory))

(defconst second-mind--modes '("memory" "kb" "hybrid")
  "Retrieval scopes accepted by the /ask endpoint.")

(defun second-mind--request (question top-k &optional mode)
  "POST QUESTION (TOP-K chunks, retrieval MODE) and return the response alist.
MODE, when non-nil and non-empty, selects the retrieval scope.  Signals a
`user-error' on transport or server failure."
  (let* ((url-request-method "POST")
         (url-request-extra-headers '(("Content-Type" . "application/json")))
         (payload (append (list (cons "question" question)
                                (cons "top_k" top-k))
                          (when (and mode (not (string-empty-p mode)))
                            (list (cons "mode" mode)))))
         (url-request-data
          (encode-coding-string (json-encode payload) 'utf-8))
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

(defun second-mind--answer-org (question top-k mode)
  "Query the endpoint and return the org-formatted answer string.
QUESTION is asked with TOP-K chunks under retrieval MODE."
  (let ((org (alist-get 'org (second-mind--request question top-k mode))))
    (if (and org (stringp org)) org "(no answer)")))

(defun second-mind--read-args ()
  "Read QUESTION and, with a prefix arg, TOP-K and MODE for a query.
Returns a list (QUESTION TOP-K MODE) suitable for `interactive'."
  (list (read-string "Second Mind — ask: ")
        (when current-prefix-arg
          (read-number "top_k: " second-mind-top-k))
        (when current-prefix-arg
          (completing-read "mode: " second-mind--modes nil t
                           nil nil second-mind-default-mode))))

;;;###autoload
(defun second-mind-ask (question &optional top-k mode)
  "Ask the local Second Mind QUESTION and show the answer with citations.
With a prefix argument, also prompt for TOP-K chunks and retrieval MODE."
  (interactive (second-mind--read-args))
  (when (string-empty-p (string-trim question))
    (user-error "Second Mind: question must not be empty"))
  (let* ((k (or top-k second-mind-top-k))
         (m (or mode second-mind-default-mode))
         (org (second-mind--answer-org question k m))
         (buffer (get-buffer-create second-mind-buffer-name)))
    (with-current-buffer buffer
      (let ((inhibit-read-only t))
        (erase-buffer)
        (org-mode)
        (insert "#+title: Second Mind\n\n")
        (insert "* Q: " question "\n\n")
        (insert org "\n"))
      (goto-char (point-min)))
    (pop-to-buffer buffer)))

;;;###autoload
(defun second-mind-insert-answer (question)
  "Query the Second Mind and insert the answer (with citations) at point.
Handy for grounding a `gptel' conversation: run this in a gptel buffer to
drop KB-cited context into the prompt before chatting."
  (interactive "sSecond Mind — insert answer for: ")
  (when (string-empty-p (string-trim question))
    (user-error "Second Mind: question must not be empty"))
  (insert (second-mind--answer-org question second-mind-top-k
                                   second-mind-default-mode)))

(defun second-mind--org-directory ()
  "Resolve the org directory `second-mind-ql' should search."
  (or second-mind-org-directory
      (bound-and-true-p org-roam-directory)
      (bound-and-true-p org-directory)
      (user-error "Second Mind: set `second-mind-org-directory'")))

;;;###autoload
(defun second-mind-ql (query)
  "Run an `org-ql' QUERY over the KB org directory.
Complements the semantic /ask path with structured, exact queries
\(tags, todo states, properties) across the same corpus.  Requires
`org-ql'."
  (interactive "sSecond Mind — org-ql query: ")
  (unless (require 'org-ql nil t)
    (user-error "Second Mind: org-ql is not installed"))
  (org-ql-search
   (directory-files-recursively (second-mind--org-directory) "\\.org\\'")
   query))

(provide 'second-mind)
;;; second-mind.el ends here
