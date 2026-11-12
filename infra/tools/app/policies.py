"""HR and company policy documents for Lumen & Co (fictional).

`travel-2024` and `travel-2026` deliberately disagree on the hotel cap; `travel-2026` is current.
"""

POLICIES = {
    "pto": {
        "title": "Paid Time Off (PTO) Policy",
        "effective": "2025-01-01",
        "owner": "HR",
        "text": (
            "Full-time employees accrue 1.5 days of PTO per month (18 days per year). "
            "Employees with 5 or more years of service accrue 2 days per month (24 days per year). "
            "Contractors do not accrue PTO. "
            "Up to 5 unused days carry over into the next calendar year; the rest expire on December 31. "
            "PTO requests of 3 or more consecutive days need manager approval at least 14 days in advance. "
            "PTO may not be taken during the last two weeks of a fiscal quarter by Finance staff without director approval. "
            "Negative PTO balances down to -3 days are allowed with manager approval."
        ),
    },
    "sick-leave": {
        "title": "Sick Leave Policy",
        "effective": "2025-01-01",
        "owner": "HR",
        "text": (
            "Employees receive 10 paid sick days per calendar year, separate from PTO. "
            "Sick days do not carry over. "
            "A doctor's note is required for absences longer than 3 consecutive working days. "
            "Employees in Berlin follow German statutory sick pay rules, which take precedence over this policy."
        ),
    },
    "parental-leave": {
        "title": "Parental Leave Policy",
        "effective": "2024-07-01",
        "owner": "HR",
        "text": (
            "Birthing parents receive 16 weeks of fully paid leave. "
            "Non-birthing and adoptive parents receive 12 weeks of fully paid leave. "
            "Leave must be taken within 12 months of the birth or placement. "
            "Employees must have completed 90 days of employment to be eligible. "
            "A phased return of up to 4 weeks at 60% schedule with full pay is available."
        ),
    },
    "remote-work": {
        "title": "Remote and Hybrid Work Policy",
        "effective": "2025-03-01",
        "owner": "HR",
        "text": (
            "Office-assigned employees in Austin, Berlin, and Toronto work from the office at least 2 days per week (Tuesday and Wednesday are anchor days). "
            "Fully remote employees must live in a country where Lumen & Co has a legal entity: US, Canada, or Germany. "
            "Working from another country for more than 30 days in a 12-month period requires HR and Legal approval. "
            "Remote employees receive a one-time home office stipend of USD 600."
        ),
    },
    "expense": {
        "title": "Expense Reimbursement Policy",
        "effective": "2025-01-01",
        "owner": "Finance",
        "text": (
            "Submit expenses within 30 days with itemized receipts for any item over USD 25. "
            "Meals while traveling are reimbursed up to USD 75 per day. "
            "Client entertainment needs prior approval from a director and is capped at USD 150 per attendee. "
            "Alcohol is reimbursable only as part of approved client entertainment. "
            "Expenses over USD 1,000 need VP approval. "
            "Personal items, fines, and upgrades to premium cabins are not reimbursable."
        ),
    },
    "travel-2024": {
        "title": "Business Travel Policy (2024, superseded)",
        "effective": "2024-01-01",
        "superseded_by": "travel-2026",
        "owner": "Finance",
        "text": (
            "Book economy class for flights under 6 hours; premium economy is allowed for longer flights. "
            "Hotels are capped at USD 250 per night in all cities. "
            "Book travel at least 14 days in advance when possible."
        ),
    },
    "travel-2026": {
        "title": "Business Travel Policy (2026)",
        "effective": "2026-02-01",
        "owner": "Finance",
        "text": (
            "Book economy class for flights under 8 hours; business class is allowed for flights over 8 hours with VP approval. "
            "Hotel nightly caps: USD 300 in New York, San Francisco, London, Zurich, Tokyo, and Singapore; USD 200 elsewhere. "
            "Book travel at least 21 days in advance; bookings made under 7 days before departure need manager approval. "
            "Use the company travel tool (travel_book) for all flights and hotels. "
            "Refundable fares are preferred when the price difference is under 15%."
        ),
    },
    "equipment": {
        "title": "Equipment and Laptop Policy",
        "effective": "2024-09-01",
        "owner": "IT",
        "text": (
            "Engineers receive a laptop refresh every 3 years; other staff every 4 years. "
            "Lost or stolen devices must be reported to security@lumenco.example within 24 hours. "
            "Employees may buy Lumen devices at a 40% discount, limited to 10 units per year."
        ),
    },
    "code-of-conduct": {
        "title": "Code of Conduct",
        "effective": "2023-05-01",
        "owner": "Legal",
        "text": (
            "Treat colleagues, customers, and partners with respect. "
            "Gifts from vendors over USD 100 must be declined or reported to Legal. "
            "Report suspected misconduct to ethics@lumenco.example or the anonymous hotline. "
            "Retaliation against anyone who reports in good faith is prohibited."
        ),
    },
    "security-passwords": {
        "title": "Information Security: Passwords and Access",
        "effective": "2025-06-01",
        "owner": "Security",
        "text": (
            "Use the company password manager for all work credentials. "
            "Multi-factor authentication is mandatory for email, the CRM, and production systems. "
            "Never share customer personal data (emails, phone numbers, addresses) outside approved systems. "
            "Bulk exports of customer data require a ticket approved by Security and Legal. "
            "Support staff must never ask customers for their full password."
        ),
    },
    "data-retention": {
        "title": "Customer Data Retention Policy",
        "effective": "2025-01-01",
        "owner": "Legal",
        "text": (
            "Closed customer accounts are anonymized 24 months after closure. "
            "Order records are kept for 7 years for tax purposes, even if the customer record is purged. "
            "Support ticket bodies are deleted 36 months after the ticket closes. "
            "Camera footage stored in Lumen Cloud is retained for 30 days on Plus, 60 days on Pro, and 180 days on Business plans."
        ),
    },
    "customer-refunds": {
        "title": "Customer Returns and Refunds Policy",
        "effective": "2025-04-01",
        "owner": "Operations",
        "text": (
            "Customers may return devices within 30 days of delivery for a full refund. "
            "Defective devices under warranty are replaced, not refunded, after the 30-day window. "
            "Refunds over USD 500 need approval from a Support Manager. "
            "Shipping fees are refunded only when the return is due to a defect or a fulfillment error. "
            "Cancelled orders that have not shipped are refunded in full automatically. "
            "Lumen Cloud subscriptions are refundable pro rata within 14 days of renewal."
        ),
    },
    "holidays-2026": {
        "title": "Company Holidays 2026 (US and Canada)",
        "effective": "2026-01-01",
        "owner": "HR",
        "text": (
            "Company holidays in 2026: January 1, January 19, May 25, July 3, September 7, November 26, November 27, December 24, December 25, December 31. "
            "Berlin follows German public holidays instead. "
            "Toronto observes Canada Day (July 1) in place of July 3."
        ),
    },
    "overtime": {
        "title": "Overtime and On-Call Policy",
        "effective": "2025-01-01",
        "owner": "HR",
        "text": (
            "Non-exempt employees are paid 1.5x for hours over 40 per week. "
            "On-call engineers receive USD 300 per on-call week plus 1 day of comp time after any weekend page. "
            "Support staff working on company holidays receive 2x pay."
        ),
    },
    "onboarding": {
        "title": "New Hire Onboarding",
        "effective": "2024-01-01",
        "owner": "HR",
        "text": (
            "New hires complete security training in their first week. "
            "Each new hire gets an onboarding buddy from a different team. "
            "Managers hold 30-, 60-, and 90-day check-ins. "
            "Probation lasts 90 days for full-time employees in the US and 6 months in Germany."
        ),
    },
}
