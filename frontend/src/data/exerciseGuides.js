/** Shared copy + step image paths for exercise picker and detail pages */

export const EXERCISE_GUIDES = {
  bird_dog: {
    phases: ["Tabletop", "Extend & hold", "Other side"],
    steps: [
      { image: "/exercises/bird_dog/step-1.png", caption: "Start on hands and knees with a flat back" },
      { image: "/exercises/bird_dog/step-2.png", caption: "Extend opposite arm and leg; hold 2–3 seconds" },
      {
        image: "/exercises/bird_dog/step-3.png",
        caption: "Switch to the other arm and leg; extend out and hold 2–3 seconds",
      },
    ],
    film: {
      title: "How to film",
      items: [
        "Side view at hip height — full body in frame",
        "Stable phone or tripod; good lighting on your mat",
        "10–20 seconds: 2–3 holds per side",
      ],
    },
    movement: {
      title: "Correct movement",
      items: [
        "Start on hands and knees, back flat",
        "Extend opposite arm and leg; hold 2–3 seconds",
        "Switch to the other arm and leg; extend and hold the same way",
        "Keep hips level — don't rotate or arch your back",
      ],
    },
    avoid: {
      title: "Common mistakes",
      items: [
        "Back sagging or over-arching",
        "Arm or leg not parallel to the floor",
        "Rushing — no clear hold at extension",
      ],
    },
  },
  bridge: {
    phases: ["Feet set", "Lift & hold", "Lower"],
    steps: [
      { image: "/exercises/bridge/step-1.png", caption: "Lie on your back, knees bent, feet hip-width" },
      { image: "/exercises/bridge/step-2.png", caption: "Lift hips until shoulders, hips, and knees align" },
      { image: "/exercises/bridge/step-3.png", caption: "Lower with control and repeat" },
    ],
    film: {
      title: "How to film",
      items: [
        "Side view — camera at hip height, full body visible",
        "Same angle every time (landscape works well)",
        "Hold the top of each rep 2–3 seconds",
      ],
    },
    movement: {
      title: "Correct movement",
      items: [
        "Lie on back, knees bent, feet hip-width apart",
        "Drive through heels; lift hips until shoulders–hips–knees align",
        "Squeeze glutes at the top; lower with control",
      ],
    },
    avoid: {
      title: "Common mistakes",
      items: [
        "Hips too low or uneven lift",
        "Knees flaring wide or feet too far from body",
        "Shoulders lifting off the mat",
      ],
    },
  },
  cat_cow: {
    phases: ["Neutral", "Cat arch", "Cow drop"],
    steps: [
      { image: "/exercises/cat_cow/step-1.png", caption: "Start on all fours — neutral spine" },
      { image: "/exercises/cat_cow/step-2.png", caption: "Cat: round your spine up, tuck chin gently" },
      { image: "/exercises/cat_cow/step-3.png", caption: "Cow: drop the belly, lift chest slightly" },
    ],
    film: {
      title: "How to film",
      items: [
        "Side view in quadruped — spine visible",
        "Hands under shoulders, knees under hips",
        "3–5 slow cycles over 15–20 seconds",
      ],
    },
    movement: {
      title: "Correct movement",
      items: [
        "On all fours; breathe with each phase",
        "Cat: round spine up, tuck chin gently",
        "Cow: drop belly, lift chest and tailbone slightly",
      ],
    },
    avoid: {
      title: "Common mistakes",
      items: [
        "Moving too fast — no clear end positions",
        "Only moving neck, not the whole spine",
        "Shifting hands/knees between reps",
      ],
    },
  },
};

export function getGuide(exerciseId) {
  return EXERCISE_GUIDES[exerciseId] ?? null;
}
