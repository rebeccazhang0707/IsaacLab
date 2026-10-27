* Added task-local arm/torso collision guarding to G1 shoelace DLS IK. Joint targets
  use bounded distance-gradient corrections and nonlinear forward-kinematics
  backtracking against arm spheres and a torso box. Actual proxy penetration beyond
  5 mm terminates the episode. Existing policies retain their observation/action
  dimensions but require re-evaluation under the changed control response.
