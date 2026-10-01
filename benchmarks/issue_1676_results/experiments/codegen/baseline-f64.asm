
/tmp/1676-baseline-bench:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001003cdee4 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E>:
1003cdee4:     	sub	sp, sp, #0x80
1003cdee8:     	stp	x28, x27, [sp, #0x20]
1003cdeec:     	stp	x26, x25, [sp, #0x30]
1003cdef0:     	stp	x24, x23, [sp, #0x40]
1003cdef4:     	stp	x22, x21, [sp, #0x50]
1003cdef8:     	stp	x20, x19, [sp, #0x60]
1003cdefc:     	stp	x29, x30, [sp, #0x70]
1003cdf00:     	add	x29, sp, #0x70
1003cdf04:     	ldr	x8, [x0, #0x10]
1003cdf08:     	ldr	x9, [x0]
1003cdf0c:     	sub	x9, x9, x8
1003cdf10:     	cmp	x2, x9
1003cdf14:     	b.hi	0x1003ce1bc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2d8>
1003cdf18:     	cbz	x3, 0x1003ce188 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2a4>
1003cdf1c:     	sub	x22, x3, #0x1
1003cdf20:     	add	x21, x2, #0x1
1003cdf24:     	cbz	x4, 0x1003ce024 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x140>
1003cdf28:     	lsl	x20, x2, #3
1003cdf2c:     	sub	x8, x4, #0x1
1003cdf30:     	cbz	x8, 0x1003ce03c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x158>
1003cdf34:     	mov	x23, #0x0               ; =0
1003cdf38:     	lsl	x24, x3, #3
1003cdf3c:     	add	x25, x1, x24
1003cdf40:     	sub	x26, x20, #0x8
1003cdf44:     	b	0x1003cdf64 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x80>
1003cdf48:     	ldr	x8, [x0, #0x10]
1003cdf4c:     	add	x23, x23, #0x1
1003cdf50:     	str	x8, [x0, #0x10]
1003cdf54:     	add	x25, x25, #0x8
1003cdf58:     	sub	x26, x26, #0x8
1003cdf5c:     	cmp	x23, x3
1003cdf60:     	b.eq	0x1003ce188 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2a4>
1003cdf64:     	cmp	x23, x21
1003cdf68:     	b.eq	0x1003ce1a8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2c4>
1003cdf6c:     	lsl	x27, x23, #3
1003cdf70:     	cmp	x20, x27
1003cdf74:     	b.eq	0x1003cdf48 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x64>
1003cdf78:     	mvn	x8, x23
1003cdf7c:     	add	x8, x2, x8
1003cdf80:     	udiv	x8, x8, x3
1003cdf84:     	add	x9, x8, #0x1
1003cdf88:     	cmp	x4, x9
1003cdf8c:     	csinc	x9, x4, x8, lo
1003cdf90:     	ldr	x8, [x0, #0x10]
1003cdf94:     	ldr	x10, [x0]
1003cdf98:     	sub	x10, x10, x8
1003cdf9c:     	cmp	x9, x10
1003cdfa0:     	b.hi	0x1003cdfe8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x104>
1003cdfa4:     	ldr	x9, [x0, #0x8]
1003cdfa8:     	ldr	d0, [x1, x27]
1003cdfac:     	str	d0, [x9, x8, lsl #3]
1003cdfb0:     	add	x8, x8, #0x1
1003cdfb4:     	mov	x10, x26
1003cdfb8:     	mov	x11, x25
1003cdfbc:     	sub	x12, x4, #0x1
1003cdfc0:     	cmp	x22, x10, lsr #3
1003cdfc4:     	b.hs	0x1003cdf4c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x68>
1003cdfc8:     	ldr	d0, [x11]
1003cdfcc:     	str	d0, [x9, x8, lsl #3]
1003cdfd0:     	add	x8, x8, #0x1
1003cdfd4:     	add	x11, x11, x24
1003cdfd8:     	sub	x10, x10, x24
1003cdfdc:     	sub	x12, x12, #0x1
1003cdfe0:     	cbnz	x12, 0x1003cdfc0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0xdc>
1003cdfe4:     	b	0x1003cdf4c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x68>
1003cdfe8:     	mov	x19, x0
1003cdfec:     	stp	x2, x1, [sp, #0x10]
1003cdff0:     	mov	x1, x8
1003cdff4:     	mov	x2, x9
1003cdff8:     	str	x3, [sp, #0x8]
1003cdffc:     	mov	w3, #0x8                ; =8
1003ce000:     	mov	x28, x4
1003ce004:     	mov	w4, #0x8                ; =8
1003ce008:     	bl	0x1028fab48 <__ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h181c4f681adaf166E>
1003ce00c:     	mov	x0, x19
1003ce010:     	ldp	x2, x1, [sp, #0x10]
1003ce014:     	mov	x4, x28
1003ce018:     	ldr	x3, [sp, #0x8]
1003ce01c:     	ldr	x8, [x19, #0x10]
1003ce020:     	b	0x1003cdfa4 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0xc0>
1003ce024:     	cmp	x22, x21
1003ce028:     	csel	x8, x22, x21, lo
1003ce02c:     	cmp	x8, #0x4
1003ce030:     	b.hs	0x1003ce100 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x21c>
1003ce034:     	mov	x8, #0x0                ; =0
1003ce038:     	b	0x1003ce120 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x23c>
1003ce03c:     	sub	x21, x2, #0x1
1003ce040:     	mov	x22, x3
1003ce044:     	b	0x1003ce064 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x180>
1003ce048:     	ldr	x8, [x0, #0x10]
1003ce04c:     	str	x8, [x0, #0x10]
1003ce050:     	sub	x20, x20, #0x8
1003ce054:     	add	x1, x1, #0x8
1003ce058:     	sub	x21, x21, #0x1
1003ce05c:     	sub	x22, x22, #0x1
1003ce060:     	cbz	x22, 0x1003ce188 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2a4>
1003ce064:     	cmn	x21, #0x2
1003ce068:     	b.eq	0x1003ce1a8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2c4>
1003ce06c:     	cbz	x20, 0x1003ce048 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x164>
1003ce070:     	udiv	x8, x21, x3
1003ce074:     	add	x9, x8, #0x1
1003ce078:     	cmp	x4, x9
1003ce07c:     	csinc	x9, x4, x8, lo
1003ce080:     	ldr	x8, [x0, #0x10]
1003ce084:     	ldr	x10, [x0]
1003ce088:     	sub	x10, x10, x8
1003ce08c:     	cmp	x9, x10
1003ce090:     	b.hi	0x1003ce0bc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x1d8>
1003ce094:     	ldr	x9, [x0, #0x8]
1003ce098:     	ldr	d0, [x1], #0x8
1003ce09c:     	str	d0, [x9, x8, lsl #3]
1003ce0a0:     	add	x8, x8, #0x1
1003ce0a4:     	str	x8, [x0, #0x10]
1003ce0a8:     	sub	x20, x20, #0x8
1003ce0ac:     	sub	x21, x21, #0x1
1003ce0b0:     	sub	x22, x22, #0x1
1003ce0b4:     	cbnz	x22, 0x1003ce064 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x180>
1003ce0b8:     	b	0x1003ce188 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2a4>
1003ce0bc:     	mov	x19, x0
1003ce0c0:     	mov	x23, x1
1003ce0c4:     	mov	x1, x8
1003ce0c8:     	mov	x24, x2
1003ce0cc:     	mov	x2, x9
1003ce0d0:     	mov	x25, x3
1003ce0d4:     	mov	w3, #0x8                ; =8
1003ce0d8:     	mov	x26, x4
1003ce0dc:     	mov	w4, #0x8                ; =8
1003ce0e0:     	bl	0x1028fab48 <__ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h181c4f681adaf166E>
1003ce0e4:     	mov	x0, x19
1003ce0e8:     	mov	x1, x23
1003ce0ec:     	mov	x4, x26
1003ce0f0:     	mov	x3, x25
1003ce0f4:     	mov	x2, x24
1003ce0f8:     	ldr	x8, [x19, #0x10]
1003ce0fc:     	b	0x1003ce094 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x1b0>
1003ce100:     	add	x8, x8, #0x1
1003ce104:     	ands	x9, x8, #0x3
1003ce108:     	mov	w10, #0x4               ; =4
1003ce10c:     	csel	x9, x10, x9, eq
1003ce110:     	sub	x8, x8, x9
1003ce114:     	mov	x9, x8
1003ce118:     	subs	x9, x9, #0x4
1003ce11c:     	b.ne	0x1003ce118 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x234>
1003ce120:     	sub	x9, x3, x8
1003ce124:     	ands	x10, x9, #0x3
1003ce128:     	b.eq	0x1003ce140 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x25c>
1003ce12c:     	cmp	x21, x8
1003ce130:     	b.eq	0x1003ce1a8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2c4>
1003ce134:     	add	x8, x8, #0x1
1003ce138:     	subs	x10, x10, #0x1
1003ce13c:     	b.ne	0x1003ce12c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x248>
1003ce140:     	sub	x9, x9, #0x1
1003ce144:     	cmp	x9, #0x3
1003ce148:     	b.lo	0x1003ce188 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2a4>
1003ce14c:     	sub	x9, x2, x8
1003ce150:     	add	x9, x9, #0x1
1003ce154:     	cbz	x9, 0x1003ce1a8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2c4>
1003ce158:     	cmp	x8, x2
1003ce15c:     	b.eq	0x1003ce1a8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2c4>
1003ce160:     	add	x10, x8, #0x1
1003ce164:     	cmp	x10, x2
1003ce168:     	b.eq	0x1003ce1a8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2c4>
1003ce16c:     	add	x10, x8, #0x2
1003ce170:     	cmp	x10, x2
1003ce174:     	b.eq	0x1003ce1a8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2c4>
1003ce178:     	add	x8, x8, #0x4
1003ce17c:     	sub	x9, x9, #0x4
1003ce180:     	cmp	x8, x3
1003ce184:     	b.ne	0x1003ce154 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x270>
1003ce188:     	ldp	x29, x30, [sp, #0x70]
1003ce18c:     	ldp	x20, x19, [sp, #0x60]
1003ce190:     	ldp	x22, x21, [sp, #0x50]
1003ce194:     	ldp	x24, x23, [sp, #0x40]
1003ce198:     	ldp	x26, x25, [sp, #0x30]
1003ce19c:     	ldp	x28, x27, [sp, #0x20]
1003ce1a0:     	add	sp, sp, #0x80
1003ce1a4:     	ret
1003ce1a8:     	adrp	x3, 0x102dfc000 <_anon.e3f3e8898f3deed588652aef9995c150.21+0x2b0>
1003ce1ac:     	add	x3, x3, #0x348
1003ce1b0:     	add	x0, x2, #0x1
1003ce1b4:     	mov	x1, x2
1003ce1b8:     	bl	0x102982ca8 <__RNvNtNtCs6sq8b9ugfBC_4core5slice5index16slice_index_fail>
1003ce1bc:     	mov	x19, x0
1003ce1c0:     	mov	x21, x1
1003ce1c4:     	mov	x1, x8
1003ce1c8:     	mov	x20, x2
1003ce1cc:     	mov	x22, x3
1003ce1d0:     	mov	w3, #0x8                ; =8
1003ce1d4:     	mov	x23, x4
1003ce1d8:     	mov	w4, #0x8                ; =8
1003ce1dc:     	bl	0x1028fab48 <__ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h181c4f681adaf166E>
1003ce1e0:     	mov	x0, x19
1003ce1e4:     	mov	x1, x21
1003ce1e8:     	mov	x4, x23
1003ce1ec:     	mov	x3, x22
1003ce1f0:     	mov	x2, x20
1003ce1f4:     	cbnz	x22, 0x1003cdf1c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x38>
1003ce1f8:     	b	0x1003ce188 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hc385901b21d70fa3E+0x2a4>
