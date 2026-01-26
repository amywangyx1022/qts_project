# Rubric Skill

Report the homework and final project expectations rubric.

## Instructions

When the user invokes `/rubric`, display the following expectations clearly:

---

## Homework and Final Project Expectations

Submissions should read like a **mini-academic paper**, telling a "story" of your research.

### Homework Guidelines

**Communication Style:**
- Be terse but include: motivation, observations/interpretations, explanations of counterintuitive findings, and speculation on meaning and extensions
- Write as if communicating with skilled colleagues abroad who cannot ask clarifying questions in real-time
- Convey ideas without being wordy - avoid LLM-generated verbosity/logorrhea
- You do NOT need to recapitulate formulas and prose from lecture notes (though you may if it aids your story)

**Code Organization:**
- **Correctness is the baseline** - code must work
- Place functions (especially long ones) in a collapsible subheading at the top for reviewer convenience
- Highlight subjective decisions (e.g., lookback periods) and provide 1-2 sentences motivating your choice

**Recommended Workflow:**
1. **Start with prose** - State what you are investigating and your rough plan
2. **Prototype your framework** - Decide what function calls you need:
   ```python
   def get_dividend_data(tickers, start_date, end_date):
       pass

   def select_tickers_with_special_dividends(all_tickers, dividend_data):
       pass

   def simulate_strategy_with_threshold(threshold, price_data, dividend_model_fits, tickers_selected, start_date, end_date):
       pass
   ```
3. **Implement data fetching first** - Plot outputs to verify sensibility
4. **Debug on smaller data subsets** - Scale up only after everything works
5. **Draw conclusions** - Complete the narrative after generating analytic output

### Final Project Expectations

Final projects should demonstrate **great communication** including:
- Clear motivation
- Explanations of what is being analyzed
- Informative and well-labeled graphics
- Professional presentation quality (reference: RavenPack-FXTradingOnNews.pdf whitepaper style)

### Key Reminders

- Tell a coherent "story" of your research
- Graphics should be informative and well-labeled
- Balance thoroughness with conciseness
- Code correctness is non-negotiable
- Document subjective choices with brief justifications

---

After displaying these expectations, offer to help the user review their current work against these criteria if they have a notebook or document to check.
