from manim import *
import numpy as np

# Problem:   minimize  f(x, y) = x^2 + 2y^2
#            subject to g(x, y) = x + y - 3 = 0
# Lagrangian: L = f + lambda * (3 - x - y)
#   dL/dx : 2x - lambda = 0
#   dL/dy : 4y - lambda = 0
#   dL/dl : 3 - x - y   = 0   ->  x* = 2, y* = 1, lambda* = 4, f* = 6


def f(x, y):
    return x**2 + 2 * y**2


class LagrangeNLP(ThreeDScene):
    def construct(self):
        # ---------------- Part 1: problem statement ----------------
        title = Text("Nonlinear Programming via Lagrange Multipliers", font_size=36)
        title.to_edge(UP)
        problem = MathTex(
            r"\min_{x,y}\; f(x,y)=x^2+2y^2 \quad\text{s.t.}\quad g(x,y)=x+y-3=0",
            font_size=38,
        ).next_to(title, DOWN, buff=0.6)
        self.add_fixed_in_frame_mobjects(title, problem)
        self.play(Write(title))
        self.play(Write(problem))
        self.wait(1)

        # ---------------- Part 2: algebra ----------------
        lagr = MathTex(
            r"\mathcal{L}(x,y,\lambda)=x^2+2y^2+\lambda\,(3-x-y)", font_size=38
        ).next_to(problem, DOWN, buff=0.6)
        stationarity = MathTex(
            r"\frac{\partial \mathcal{L}}{\partial x}=2x-\lambda=0,\quad"
            r"\frac{\partial \mathcal{L}}{\partial y}=4y-\lambda=0,\quad"
            r"\frac{\partial \mathcal{L}}{\partial \lambda}=3-x-y=0",
            font_size=32,
        ).next_to(lagr, DOWN, buff=0.5)
        solve1 = MathTex(r"x=\tfrac{\lambda}{2},\;\; y=\tfrac{\lambda}{4}\;\Rightarrow\;"
                         r"\tfrac{\lambda}{2}+\tfrac{\lambda}{4}=3",
                         font_size=34).next_to(stationarity, DOWN, buff=0.5)
        solution = MathTex(
            r"\lambda^*=4,\quad x^*=2,\quad y^*=1,\quad f(x^*,y^*)=6",
            font_size=38, color=YELLOW,
        ).next_to(solve1, DOWN, buff=0.5)
        for m in (lagr, stationarity, solve1, solution):
            self.add_fixed_in_frame_mobjects(m)
            self.remove(m)
        self.play(Write(lagr))
        self.wait(0.5)
        self.play(Write(stationarity))
        self.wait(0.5)
        self.play(Write(solve1))
        self.wait(0.5)
        self.play(Write(solution))
        self.wait(2)
        self.play(*[FadeOut(m) for m in (lagr, stationarity, solve1, problem)])

        # ---------------- Part 3: 3D surface ----------------
        axes = ThreeDAxes(
            x_range=[-1, 4, 1], y_range=[-1, 3, 1], z_range=[0, 36, 6],
            x_length=6, y_length=4.5, z_length=4,
        )
        labels = VGroup(
            axes.get_x_axis_label("x"),
            axes.get_y_axis_label("y"),
            axes.get_z_axis_label("f"),
        )

        surface = Surface(
            lambda u, v: axes.c2p(u, v, f(u, v)),
            u_range=[-1, 4], v_range=[-1, 3],
            resolution=(36, 36),
            fill_opacity=0.55,
            checkerboard_colors=[BLUE_D, BLUE_E],
            stroke_width=0.4, stroke_color=BLUE_B,
        )

        sub_title = Text("Objective surface  f(x,y) = x² + 2y²", font_size=28)
        sub_title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(sub_title)
        self.remove(sub_title)

        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES, zoom=0.85)
        self.play(FadeOut(title), Create(axes), FadeIn(labels), run_time=1.5)
        self.play(Create(surface), Write(sub_title), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(2)

        # Unconstrained minimum at the origin
        origin_pt = Dot3D(axes.c2p(0, 0, 0), color=GREEN, radius=0.09)
        origin_lbl = MathTex(r"\text{unconstrained min } (0,0)", font_size=26, color=GREEN)
        origin_lbl.to_corner(DL)
        self.add_fixed_in_frame_mobjects(origin_lbl)
        self.remove(origin_lbl)
        self.play(FadeIn(origin_pt), Write(origin_lbl))
        self.wait(1.5)

        # ---------------- Part 4: feasible region & cut ----------------
        new_sub = Text("Feasible set  x + y = 3  cuts the surface", font_size=28).to_corner(UL)
        self.add_fixed_in_frame_mobjects(new_sub)
        self.remove(new_sub)

        # Feasible line in the x-y plane
        feas_line = ParametricFunction(
            lambda t: axes.c2p(t, 3 - t, 0), t_range=[0, 4], color=ORANGE, stroke_width=6
        )
        # Vertical plane (the "cut") through the feasible line
        cut_plane = Surface(
            lambda u, w: axes.c2p(u, 3 - u, w),
            u_range=[0, 4], v_range=[0, 34],
            resolution=(8, 8),
            fill_color=ORANGE, fill_opacity=0.25,
            stroke_width=0.6, stroke_color=ORANGE,
        )
        # Curve where the plane meets the surface: f restricted to the constraint
        cut_curve = ParametricFunction(
            lambda t: axes.c2p(t, 3 - t, f(t, 3 - t)),
            t_range=[0, 4], color=YELLOW, stroke_width=7,
        )

        self.play(Transform(sub_title, new_sub), run_time=1)
        self.play(Create(feas_line), run_time=1.5)
        self.play(FadeIn(cut_plane), run_time=1.5)
        self.play(Create(cut_curve), run_time=2)
        self.wait(1)

        # Slide a point along the curve to show the 1-D profile, settle at the min
        t_tracker = ValueTracker(0.2)
        mover = always_redraw(
            lambda: Dot3D(
                axes.c2p(t_tracker.get_value(), 3 - t_tracker.get_value(),
                         f(t_tracker.get_value(), 3 - t_tracker.get_value())),
                color=RED, radius=0.09,
            )
        )
        foot = always_redraw(
            lambda: Dot3D(
                axes.c2p(t_tracker.get_value(), 3 - t_tracker.get_value(), 0),
                color=RED, radius=0.06,
            )
        )
        drop = always_redraw(
            lambda: DashedLine(foot.get_center(), mover.get_center(), color=RED)
        )
        self.add(mover, foot, drop)
        self.play(t_tracker.animate.set_value(3.8), run_time=3, rate_func=smooth)
        self.play(t_tracker.animate.set_value(2.0), run_time=2.5, rate_func=smooth)
        self.wait(0.5)

        # Mark the constrained minimum
        star = Dot3D(axes.c2p(2, 1, 6), color=RED, radius=0.14)
        res_lbl = MathTex(
            r"\text{constrained local min: } (x^*,y^*)=(2,1),\; f^*=6",
            font_size=30, color=RED,
        ).to_corner(DR)
        self.add_fixed_in_frame_mobjects(res_lbl)
        self.remove(res_lbl)
        self.play(FadeIn(star, scale=2), Write(res_lbl))
        self.wait(1)

        # ---------------- Part 5: gradient condition ----------------
        # grad f = (2x, 4y) = (4, 4) at optimum;  grad g = (1, 1)  ->  grad f = 4 * grad g
        grad_f = Arrow3D(axes.c2p(2, 1, 0), axes.c2p(2 + 0.9, 1 + 0.9, 0), color=PURPLE)
        grad_g = Arrow3D(axes.c2p(2, 1, 0), axes.c2p(2 + 0.5, 1 + 0.5, 0), color=TEAL)
        cond = MathTex(
            r"\nabla f(x^*)=\lambda^*\nabla g(x^*)\;\;\Rightarrow\;\;(4,4)=4\,(1,1)",
            font_size=30, color=PURPLE,
        ).to_corner(UR)
        self.add_fixed_in_frame_mobjects(cond)
        self.remove(cond)
        self.play(Create(grad_f), Create(grad_g), Write(cond))
        self.wait(4)

        self.stop_ambient_camera_rotation()
        self.wait(1)
