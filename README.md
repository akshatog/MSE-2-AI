# KIET_AI_DebugChallenge

# Bugs
Bug 2 → Crash immediately: vocabulary is empty since all text tokens are letters, never digits

Bug 3 + 4 → Training runs but loss never decreases; accuracy stays near random (model gets zero-vector inputs and wrong gradients)

Bug 5 → Training looks successful, but during chat the bot always says "I don't understand" because threshold 0.95 is almost never reached

Bug 1 → Even after fixing bugs 2–5, the bot still refuses every response — students must find the config value 

The bugs are ordered by discovery difficulty, forcing students to debug in layers: crash → training failure → inference failure.

## Solution / Fixes Applied

I successfully found and resolved all 5 intentional bugs in the codebase:

1. **Bug 1 (Confidence Threshold):** 
   - **Fix:** Lowered `CONFIDENCE_THRESHOLD` from `0.95` to `0.5` so the bot provides responses instead of constantly hitting the fallback threshold.
2. **Bug 2 (Empty Vocabulary):** 
   - **Fix:** Changed `tok.isdigit()` to `tok.isalpha()` in `preprocess()` to correctly keep alphabetical words instead of filtering them out.
3. **Bug 3 (Zero Vectors):** 
   - **Fix:** Changed `vec[idx] = 0.0` to `vec[idx] = 1.0` in `_one_hot()` so the target classes are correctly one-hot encoded instead of being all zeros.
4. **Bug 4 (Wrong Math Derivative):** 
   - **Fix:** Corrected the tanh derivative in `backward()` from `(1.0 + h^2)` to `(1.0 - h^2)` to ensure gradients flow correctly and loss decreases.
5. **Bug 5 (Softmax Overflow):** 
   - **Fix:** Subtracted `x.max()` inside the `_softmax()` exponentiation (`np.exp(x - x.max())`) to prevent numerical overflow (`nan`/`inf` issues).

The bot now successfully trains and can chat accurately using the intents!
