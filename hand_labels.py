#!/usr/bin/env python3
"""Hand-assigned labels for a 50-tweet subset (taken from golden_eval.csv).

Provenance: assigned by a human reviewer reading each tweet + the author's
observed history.  Intent is what the customer is ACTUALLY about (not the silver
topic label); route is whether a human support lead would auto-handle (safe,
templated) vs escalate.  `other` marks tweets 5-class intent can't capture
(gibberish / off-topic).  Used by build_golden_hand.py.
"""
HAND_LABELS = {
    # tweet_id: (hand_intent, hand_route)
    "2455253": ("other", "assist"),        # gibberish
    "515270": ("delivery_issue", "assist"),
    "757718": ("order_status", "assist"),  # schedule pickup (return)
    "1299400": ("customer_service", "assist"),  # vague complaint, mislabeled appreciation
    "348221": ("customer_service", "assist"),   # escalating complaint
    "374157": ("customer_service", "assist"),   # truncated thread filler
    "1303634": ("customer_service", "assist"),  # generic plea
    "288034": ("delivery_issue", "assist"),     # receipt/delivery discrepancy
    "1653422": ("customer_service", "assist"),  # vague follow-up
    "962520": ("customer_service", "auto"),     # echo-dot setup -> templated how-to
    "959294": ("customer_service", "assist"),
    "324244": ("customer_service", "assist"),   # prime escalation
    "2514674": ("customer_service", "assist"),  # loss compensation
    "280148": ("customer_service", "assist"),   # account lock complaint
    "226339": ("customer_service", "assist"),   # consumer-court threat
    "1072550": ("order_status", "assist"),      # stolen-money refund
    "2883304": ("email_contact", "assist"),     # no phone contact option
    "2346213": ("order_status", "assist"),      # return/courier dispute
    "2355189": ("email_contact", "assist"),     # fake email
    "2078539": ("other", "assist"),             # Spanish, off-topic
    "714749": ("delivery_issue", "assist"),
    "222687": ("delivery_issue", "auto"),       # carrier feedback -> ack
    "1449155": ("order_status", "auto"),        # cancel Prime -> templated
    "2440783": ("delivery_issue", "assist"),
    "1498542": ("delivery_issue", "assist"),
    "93672": ("delivery_issue", "assist"),
    "1229471": ("delivery_issue", "assist"),    # prime-now availability
    "233785": ("delivery_issue", "assist"),     # tracking mismatch
    "2400339": ("appreciation", "auto"),        # praise of delivery
    "1219174": ("delivery_issue", "assist"),
    "1983560": ("email_contact", "assist"),     # email-verification expiry
    "2184547": ("customer_service", "assist"),  # account security concern
    "2302234": ("order_status", "assist"),      # refund wrong amount
    "665640": ("delivery_issue", "assist"),
    "779871": ("delivery_issue", "assist"),
    "1570890": ("email_contact", "auto"),       # providing email id -> ack
    "1100564": ("email_contact", "assist"),     # sharing details, follow-up needed
    "1721485": ("customer_service", "assist"),
    "1732078": ("email_contact", "assist"),     # can't login to contact
    "1305186": ("delivery_issue", "assist"),
    "574928": ("order_status", "assist"),       # no ship date yet
    "785125": ("order_status", "assist"),       # gift must arrive before oct
    "1490808": ("order_status", "assist"),      # unresolved oct order
    "1843616": ("order_status", "assist"),      # cancel all pending orders
    "1302268": ("order_status", "assist"),      # refunded + product with me
    "2526919": ("order_status", "assist"),      # expedite shipment
    "2171450": ("order_status", "assist"),      # confirm vs cancel mismatch
    "467390": ("email_contact", "assist"),      # share detail thread
    "1445305": ("order_status", "auto"),        # cancel audible book -> templated
    "2619642": ("order_status", "assist"),      # next-day shipping failed
}    # end HAND_LABELS