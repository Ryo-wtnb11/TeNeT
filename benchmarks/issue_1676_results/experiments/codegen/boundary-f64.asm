
/tmp/1676-boundary-bench:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100219d28 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E>:
100219d28:     	sub	sp, sp, #0xb0
100219d2c:     	stp	x28, x27, [sp, #0x50]
100219d30:     	stp	x26, x25, [sp, #0x60]
100219d34:     	stp	x24, x23, [sp, #0x70]
100219d38:     	stp	x22, x21, [sp, #0x80]
100219d3c:     	stp	x20, x19, [sp, #0x90]
100219d40:     	stp	x29, x30, [sp, #0xa0]
100219d44:     	add	x29, sp, #0xa0
100219d48:     	mov	x21, x4
100219d4c:     	mov	x19, x3
100219d50:     	mov	x22, x1
100219d54:     	mov	x23, x0
100219d58:     	mov	x24, x0
100219d5c:     	ldr	x1, [x24, #0x10]!
100219d60:     	ldr	x8, [x0]
100219d64:     	sub	x8, x8, x1
100219d68:     	cmp	x2, x8
100219d6c:     	b.hi	0x100219f14 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x1ec>
100219d70:     	str	x2, [sp]
100219d74:     	cbz	x19, 0x100219edc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x1b4>
100219d78:     	sub	x10, x19, #0x1
100219d7c:     	ldr	x8, [sp]
100219d80:     	add	x9, x8, #0x1
100219d84:     	cbz	x21, 0x100219e2c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x104>
100219d88:     	lsl	x26, x8, #3
100219d8c:     	add	x27, x22, x26
100219d90:     	add	x28, sp, #0x8
100219d94:     	sub	x25, x8, #0x1
100219d98:     	mov	x20, x19
100219d9c:     	b	0x100219de4 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0xbc>
100219da0:     	ldr	x1, [x24]
100219da4:     	ldr	x8, [x23, #0x8]
100219da8:     	stp	x21, x22, [sp, #0x8]
100219dac:     	sub	x9, x19, #0x1
100219db0:     	stp	x27, x9, [sp, #0x18]
100219db4:     	mov	w9, #0x1                ; =1
100219db8:     	strb	w9, [sp, #0x28]
100219dbc:     	stp	x28, x24, [sp, #0x30]
100219dc0:     	stp	x1, x8, [sp, #0x40]
100219dc4:     	add	x0, x28, #0x8
100219dc8:     	add	x1, sp, #0x30
100219dcc:     	bl	0x10038f78c <__ZN105_$LT$core..iter..adapters..step_by..StepBy$LT$I$GT$$u20$as$u20$core..iter..traits..iterator..Iterator$GT$8try_fold17hb05f92851a262f9eE>
100219dd0:     	sub	x26, x26, #0x8
100219dd4:     	add	x22, x22, #0x8
100219dd8:     	sub	x25, x25, #0x1
100219ddc:     	sub	x20, x20, #0x1
100219de0:     	cbz	x20, 0x100219edc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x1b4>
100219de4:     	cmn	x25, #0x2
100219de8:     	b.eq	0x100219efc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x1d4>
100219dec:     	cbz	x26, 0x100219da0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x78>
100219df0:     	udiv	x8, x25, x19
100219df4:     	add	x9, x8, #0x1
100219df8:     	cmp	x21, x9
100219dfc:     	csinc	x2, x21, x8, lo
100219e00:     	ldr	x1, [x23, #0x10]
100219e04:     	ldr	x8, [x23]
100219e08:     	sub	x8, x8, x1
100219e0c:     	cmp	x2, x8
100219e10:     	b.ls	0x100219da4 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x7c>
100219e14:     	mov	x0, x23
100219e18:     	mov	w3, #0x8                ; =8
100219e1c:     	mov	w4, #0x8                ; =8
100219e20:     	bl	0x102908834 <__ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h181c4f681adaf166E>
100219e24:     	ldr	x1, [x23, #0x10]
100219e28:     	b	0x100219da4 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x7c>
100219e2c:     	cmp	x10, x9
100219e30:     	csel	x8, x10, x9, lo
100219e34:     	cmp	x8, #0x4
100219e38:     	b.hs	0x100219e44 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x11c>
100219e3c:     	mov	x8, #0x0                ; =0
100219e40:     	b	0x100219e64 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x13c>
100219e44:     	add	x8, x8, #0x1
100219e48:     	ands	x10, x8, #0x3
100219e4c:     	mov	w11, #0x4               ; =4
100219e50:     	csel	x10, x11, x10, eq
100219e54:     	sub	x8, x8, x10
100219e58:     	mov	x10, x8
100219e5c:     	subs	x10, x10, #0x4
100219e60:     	b.ne	0x100219e5c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x134>
100219e64:     	sub	x10, x19, x8
100219e68:     	ands	x11, x10, #0x3
100219e6c:     	b.eq	0x100219e84 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x15c>
100219e70:     	cmp	x9, x8
100219e74:     	b.eq	0x100219efc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x1d4>
100219e78:     	add	x8, x8, #0x1
100219e7c:     	subs	x11, x11, #0x1
100219e80:     	b.ne	0x100219e70 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x148>
100219e84:     	sub	x9, x10, #0x1
100219e88:     	cmp	x9, #0x3
100219e8c:     	b.lo	0x100219edc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x1b4>
100219e90:     	ldr	x9, [sp]
100219e94:     	sub	x9, x9, x8
100219e98:     	add	x9, x9, #0x1
100219e9c:     	cbz	x9, 0x100219efc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x1d4>
100219ea0:     	ldr	x10, [sp]
100219ea4:     	cmp	x8, x10
100219ea8:     	b.eq	0x100219efc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x1d4>
100219eac:     	add	x10, x8, #0x1
100219eb0:     	ldr	x11, [sp]
100219eb4:     	cmp	x10, x11
100219eb8:     	b.eq	0x100219efc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x1d4>
100219ebc:     	add	x10, x8, #0x2
100219ec0:     	ldr	x11, [sp]
100219ec4:     	cmp	x10, x11
100219ec8:     	b.eq	0x100219efc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x1d4>
100219ecc:     	add	x8, x8, #0x4
100219ed0:     	sub	x9, x9, #0x4
100219ed4:     	cmp	x8, x19
100219ed8:     	b.ne	0x100219e9c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x174>
100219edc:     	ldp	x29, x30, [sp, #0xa0]
100219ee0:     	ldp	x20, x19, [sp, #0x90]
100219ee4:     	ldp	x22, x21, [sp, #0x80]
100219ee8:     	ldp	x24, x23, [sp, #0x70]
100219eec:     	ldp	x26, x25, [sp, #0x60]
100219ef0:     	ldp	x28, x27, [sp, #0x50]
100219ef4:     	add	sp, sp, #0xb0
100219ef8:     	ret
100219efc:     	adrp	x3, 0x102e01000 <_anon.e4b964cfbeeb51613e317372ed9890db.30+0x6f8>
100219f00:     	add	x3, x3, #0x720
100219f04:     	ldr	x1, [sp]
100219f08:     	add	x0, x1, #0x1
100219f0c:     	mov	x2, x1
100219f10:     	bl	0x10298a514 <__RNvNtNtCs6sq8b9ugfBC_4core5slice5index16slice_index_fail>
100219f14:     	mov	x0, x23
100219f18:     	mov	x20, x2
100219f1c:     	mov	w3, #0x8                ; =8
100219f20:     	mov	w4, #0x8                ; =8
100219f24:     	bl	0x102908834 <__ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h181c4f681adaf166E>
100219f28:     	str	x20, [sp]
100219f2c:     	cbnz	x19, 0x100219d78 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x50>
100219f30:     	b	0x100219edc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x1b4>
