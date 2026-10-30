// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): the game's camera, hooked where the engine sets it up so every pass it derives (culling, lighting,
// motion vectors) follows the head: the player's view gets the head pose and the headset's field of view. Uncharted 1:
// the views' projection update (vr_camera.cpp Sites); Driveclub's view setters are in the Driveclub recomp's vr branch.

#pragma once

#include "common/types.h"

namespace Vulkan::Vr::Camera {

/// Installs the hooks once the game image is in memory (checks the setters' bytes first). Safe to call every frame.
void Install();

/// The head pose to apply (row-vector 4x4: rows are the head's axes and position in the recentred tracking space,
/// metres), and the field of view to render (tangents of the half angles). From the frame loop.
/// `layer_pose` (orientation x, y, z, w, position x, y, z): the same head in the headset layer's space.
void SetHead(const float (&pose)[16], float tan_left, float tan_right, float tan_up, float tan_down,
             const float (&layer_pose)[7]);

/// The player camera's near plane (reversed infinite projection: P[3][2]), 0 before the first frame.
float Near();

/// Development: logs the next `count` setter calls (view, projection, position).
void LogNext(int count);

/// Development: the recent rendered camera positions (newest last), at most `max`.
u32 RecentPositions(float (*positions)[3], u32 max);

/// The game's screen aspect (the HUD's canvas; the render size may differ: a square render keeps 16:9 here).
float ScreenAspect();
/// The game's own horizontal projection scale of the main view this frame (U1: 1.30 in play and the pause menu,
/// 2.41 aiming): the aim zoom tells aiming (the reticle layer). 0 before the first view.
float GameXScale();

/// Head aim (BB_VR_HEAD_AIM=1), while it holds the view still: the aim camera's direction in the view's reference frame (degrees right, up),
/// where the reticle goes. False as above.
bool AimOffset(float& right, float& up);

/// The pose and field of view the last player view was rendered with (for the headset layer).
bool LastRendered(float (&layer_pose)[7], float (&tangents)[4]);

/// Uncharted: the pose and field of view of the recent player view whose tagged near plane (as the shaders get it) is
/// `shader_near`, else whose depth term (-near * far / (far - near)) is `depth_term`. False when neither matches.
bool TaggedRendered(float shader_near, float depth_term, float (&layer_pose)[7], float (&tangents)[4]);

/// The pose and field of view of the recent player view rendered from `position` (the camera's world position as the GPU
/// sees it, the head's offset included); `distance` gets the closest match's distance.
bool RenderedAt(const float* position, float (&layer_pose)[7], float (&tangents)[4], float& distance);

} // namespace Vulkan::Vr::Camera
