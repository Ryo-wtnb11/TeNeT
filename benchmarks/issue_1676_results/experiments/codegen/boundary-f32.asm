
/tmp/1676-boundary-bench:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100219f34 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E>:
100219f34:     	sub	sp, sp, #0xb0
100219f38:     	stp	x28, x27, [sp, #0x50]
100219f3c:     	stp	x26, x25, [sp, #0x60]
100219f40:     	stp	x24, x23, [sp, #0x70]
100219f44:     	stp	x22, x21, [sp, #0x80]
100219f48:     	stp	x20, x19, [sp, #0x90]
100219f4c:     	stp	x29, x30, [sp, #0xa0]
100219f50:     	add	x29, sp, #0xa0
100219f54:     	mov	x21, x4
100219f58:     	mov	x19, x3
100219f5c:     	mov	x22, x1
100219f60:     	mov	x23, x0
100219f64:     	mov	x24, x0
100219f68:     	ldr	x1, [x24, #0x10]!
100219f6c:     	ldr	x8, [x0]
100219f70:     	sub	x8, x8, x1
100219f74:     	cmp	x2, x8
100219f78:     	b.hi	0x10021a120 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x1ec>
100219f7c:     	str	x2, [sp]
100219f80:     	cbz	x19, 0x10021a0e8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x1b4>
100219f84:     	sub	x10, x19, #0x1
100219f88:     	ldr	x8, [sp]
100219f8c:     	add	x9, x8, #0x1
100219f90:     	cbz	x21, 0x10021a038 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x104>
100219f94:     	lsl	x26, x8, #2
100219f98:     	add	x27, x22, x26
100219f9c:     	add	x28, sp, #0x8
100219fa0:     	sub	x25, x8, #0x1
100219fa4:     	mov	x20, x19
100219fa8:     	b	0x100219ff0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0xbc>
100219fac:     	ldr	x1, [x24]
100219fb0:     	ldr	x8, [x23, #0x8]
100219fb4:     	stp	x21, x22, [sp, #0x8]
100219fb8:     	sub	x9, x19, #0x1
100219fbc:     	stp	x27, x9, [sp, #0x18]
100219fc0:     	mov	w9, #0x1                ; =1
100219fc4:     	strb	w9, [sp, #0x28]
100219fc8:     	stp	x28, x24, [sp, #0x30]
100219fcc:     	stp	x1, x8, [sp, #0x40]
100219fd0:     	add	x0, x28, #0x8
100219fd4:     	add	x1, sp, #0x30
100219fd8:     	bl	0x10038f6ac <__ZN105_$LT$core..iter..adapters..step_by..StepBy$LT$I$GT$$u20$as$u20$core..iter..traits..iterator..Iterator$GT$8try_fold17h809be07312e55d80E>
100219fdc:     	sub	x26, x26, #0x4
100219fe0:     	add	x22, x22, #0x4
100219fe4:     	sub	x25, x25, #0x1
100219fe8:     	sub	x20, x20, #0x1
100219fec:     	cbz	x20, 0x10021a0e8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x1b4>
100219ff0:     	cmn	x25, #0x2
100219ff4:     	b.eq	0x10021a108 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x1d4>
100219ff8:     	cbz	x26, 0x100219fac <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x78>
100219ffc:     	udiv	x8, x25, x19
10021a000:     	add	x9, x8, #0x1
10021a004:     	cmp	x21, x9
10021a008:     	csinc	x2, x21, x8, lo
10021a00c:     	ldr	x1, [x23, #0x10]
10021a010:     	ldr	x8, [x23]
10021a014:     	sub	x8, x8, x1
10021a018:     	cmp	x2, x8
10021a01c:     	b.ls	0x100219fb0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x7c>
10021a020:     	mov	x0, x23
10021a024:     	mov	w3, #0x4                ; =4
10021a028:     	mov	w4, #0x4                ; =4
10021a02c:     	bl	0x102908834 <__ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h181c4f681adaf166E>
10021a030:     	ldr	x1, [x23, #0x10]
10021a034:     	b	0x100219fb0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x7c>
10021a038:     	cmp	x10, x9
10021a03c:     	csel	x8, x10, x9, lo
10021a040:     	cmp	x8, #0x4
10021a044:     	b.hs	0x10021a050 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x11c>
10021a048:     	mov	x8, #0x0                ; =0
10021a04c:     	b	0x10021a070 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x13c>
10021a050:     	add	x8, x8, #0x1
10021a054:     	ands	x10, x8, #0x3
10021a058:     	mov	w11, #0x4               ; =4
10021a05c:     	csel	x10, x11, x10, eq
10021a060:     	sub	x8, x8, x10
10021a064:     	mov	x10, x8
10021a068:     	subs	x10, x10, #0x4
10021a06c:     	b.ne	0x10021a068 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x134>
10021a070:     	sub	x10, x19, x8
10021a074:     	ands	x11, x10, #0x3
10021a078:     	b.eq	0x10021a090 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x15c>
10021a07c:     	cmp	x9, x8
10021a080:     	b.eq	0x10021a108 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x1d4>
10021a084:     	add	x8, x8, #0x1
10021a088:     	subs	x11, x11, #0x1
10021a08c:     	b.ne	0x10021a07c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x148>
10021a090:     	sub	x9, x10, #0x1
10021a094:     	cmp	x9, #0x3
10021a098:     	b.lo	0x10021a0e8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x1b4>
10021a09c:     	ldr	x9, [sp]
10021a0a0:     	sub	x9, x9, x8
10021a0a4:     	add	x9, x9, #0x1
10021a0a8:     	cbz	x9, 0x10021a108 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x1d4>
10021a0ac:     	ldr	x10, [sp]
10021a0b0:     	cmp	x8, x10
10021a0b4:     	b.eq	0x10021a108 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x1d4>
10021a0b8:     	add	x10, x8, #0x1
10021a0bc:     	ldr	x11, [sp]
10021a0c0:     	cmp	x10, x11
10021a0c4:     	b.eq	0x10021a108 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x1d4>
10021a0c8:     	add	x10, x8, #0x2
10021a0cc:     	ldr	x11, [sp]
10021a0d0:     	cmp	x10, x11
10021a0d4:     	b.eq	0x10021a108 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x1d4>
10021a0d8:     	add	x8, x8, #0x4
10021a0dc:     	sub	x9, x9, #0x4
10021a0e0:     	cmp	x8, x19
10021a0e4:     	b.ne	0x10021a0a8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x174>
10021a0e8:     	ldp	x29, x30, [sp, #0xa0]
10021a0ec:     	ldp	x20, x19, [sp, #0x90]
10021a0f0:     	ldp	x22, x21, [sp, #0x80]
10021a0f4:     	ldp	x24, x23, [sp, #0x70]
10021a0f8:     	ldp	x26, x25, [sp, #0x60]
10021a0fc:     	ldp	x28, x27, [sp, #0x50]
10021a100:     	add	sp, sp, #0xb0
10021a104:     	ret
10021a108:     	adrp	x3, 0x102e01000 <_anon.e4b964cfbeeb51613e317372ed9890db.30+0x6f8>
10021a10c:     	add	x3, x3, #0x720
10021a110:     	ldr	x1, [sp]
10021a114:     	add	x0, x1, #0x1
10021a118:     	mov	x2, x1
10021a11c:     	bl	0x10298a514 <__RNvNtNtCs6sq8b9ugfBC_4core5slice5index16slice_index_fail>
10021a120:     	mov	x0, x23
10021a124:     	mov	x20, x2
10021a128:     	mov	w3, #0x4                ; =4
10021a12c:     	mov	w4, #0x4                ; =4
10021a130:     	bl	0x102908834 <__ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h181c4f681adaf166E>
10021a134:     	str	x20, [sp]
10021a138:     	cbnz	x19, 0x100219f84 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x50>
10021a13c:     	b	0x10021a0e8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x1b4>
