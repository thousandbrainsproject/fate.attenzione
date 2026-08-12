# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Constants shared across the Monty framework."""

MAX_PERCEPT_DISTANCE = 100.0
"""Distance, in meters, at or beyond which a sensed surface is off-object.

This is the single source of the perceptual range convention:
`MissingToMaxDepth` writes this value into pixels where the sensor saw
nothing, `DepthTo3DLocations` marks any pixel at or beyond it as off-object,
motor policies treat a center depth at or beyond it as having fallen off the
object, and depth salience normalizes by it so that the background void
carries zero salience.
"""
