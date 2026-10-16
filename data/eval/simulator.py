"""Reference simulated user (the only code allowed to read hidden_slots, resolutions and latent_reveal)."""
LEVEL = lambda z: "pos" if z >= 1 else "pos_mild" if z >= 0.4 else "dontmind" if z > -0.4 else "neg_mild" if z > -0.8 else "neg"
NAME = {"R": "Realistic", "I": "Investigative", "A": "Artistic", "S": "Social", "E": "Enterprising", "C": "Conventional"}

class SimulatedUser:
    def __init__(self, profile, bank):
        self.p, self.bank = profile, bank
    def opening(self):
        return self.p["text"]
    def answer(self, action, slot=None, variant=0):
        """action: CLARIFY / CHALLENGE / PROBE / ABSTAIN / RECOMMEND.
        slot: one letter R..C (convert full names with LET = {v: k for k, v in NAME.items()}).
        variant: index of the CLARIFY phrasing the bot used (wraps at 3)."""
        if action == "CLARIFY":      # five answer levels chosen by the hidden score
            return self.bank["ANSWERS"][NAME[slot]][variant % 3][LEVEL(self.p["hidden_slots"][slot])]
        if action == "CHALLENGE":    # which of my two statements is closer
            return self.p["resolutions"][slot]
        if action == "PROBE":        # only a PROBE ever receives the reveal; profiles without one restate their view
            return self.p["latent_reveal"] or self.p["resolutions"][slot]
        if action == "ABSTAIN":      # the bot rephrased with another CLARIFY variant for the same slot
            if slot: return self.answer("CLARIFY", slot, variant)
            return "sorry, what do you mean?"
        return ""
