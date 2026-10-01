
/tmp/1676-baseline-bench:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001003ce1fc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E>:
1003ce1fc:     	sub	sp, sp, #0x80
1003ce200:     	stp	x28, x27, [sp, #0x20]
1003ce204:     	stp	x26, x25, [sp, #0x30]
1003ce208:     	stp	x24, x23, [sp, #0x40]
1003ce20c:     	stp	x22, x21, [sp, #0x50]
1003ce210:     	stp	x20, x19, [sp, #0x60]
1003ce214:     	stp	x29, x30, [sp, #0x70]
1003ce218:     	add	x29, sp, #0x70
1003ce21c:     	ldr	x8, [x0, #0x10]
1003ce220:     	ldr	x9, [x0]
1003ce224:     	sub	x9, x9, x8
1003ce228:     	cmp	x2, x9
1003ce22c:     	b.hi	0x1003ce4d4 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2d8>
1003ce230:     	cbz	x3, 0x1003ce4a0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2a4>
1003ce234:     	sub	x22, x3, #0x1
1003ce238:     	add	x21, x2, #0x1
1003ce23c:     	cbz	x4, 0x1003ce33c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x140>
1003ce240:     	lsl	x20, x2, #2
1003ce244:     	sub	x8, x4, #0x1
1003ce248:     	cbz	x8, 0x1003ce354 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x158>
1003ce24c:     	mov	x23, #0x0               ; =0
1003ce250:     	lsl	x24, x3, #2
1003ce254:     	add	x25, x1, x24
1003ce258:     	sub	x26, x20, #0x4
1003ce25c:     	b	0x1003ce27c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x80>
1003ce260:     	ldr	x8, [x0, #0x10]
1003ce264:     	add	x23, x23, #0x1
1003ce268:     	str	x8, [x0, #0x10]
1003ce26c:     	add	x25, x25, #0x4
1003ce270:     	sub	x26, x26, #0x4
1003ce274:     	cmp	x23, x3
1003ce278:     	b.eq	0x1003ce4a0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2a4>
1003ce27c:     	cmp	x23, x21
1003ce280:     	b.eq	0x1003ce4c0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2c4>
1003ce284:     	lsl	x27, x23, #2
1003ce288:     	cmp	x20, x27
1003ce28c:     	b.eq	0x1003ce260 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x64>
1003ce290:     	mvn	x8, x23
1003ce294:     	add	x8, x2, x8
1003ce298:     	udiv	x8, x8, x3
1003ce29c:     	add	x9, x8, #0x1
1003ce2a0:     	cmp	x4, x9
1003ce2a4:     	csinc	x9, x4, x8, lo
1003ce2a8:     	ldr	x8, [x0, #0x10]
1003ce2ac:     	ldr	x10, [x0]
1003ce2b0:     	sub	x10, x10, x8
1003ce2b4:     	cmp	x9, x10
1003ce2b8:     	b.hi	0x1003ce300 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x104>
1003ce2bc:     	ldr	x9, [x0, #0x8]
1003ce2c0:     	ldr	s0, [x1, x27]
1003ce2c4:     	str	s0, [x9, x8, lsl #2]
1003ce2c8:     	add	x8, x8, #0x1
1003ce2cc:     	mov	x10, x26
1003ce2d0:     	mov	x11, x25
1003ce2d4:     	sub	x12, x4, #0x1
1003ce2d8:     	cmp	x22, x10, lsr #2
1003ce2dc:     	b.hs	0x1003ce264 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x68>
1003ce2e0:     	ldr	s0, [x11]
1003ce2e4:     	str	s0, [x9, x8, lsl #2]
1003ce2e8:     	add	x8, x8, #0x1
1003ce2ec:     	add	x11, x11, x24
1003ce2f0:     	sub	x10, x10, x24
1003ce2f4:     	sub	x12, x12, #0x1
1003ce2f8:     	cbnz	x12, 0x1003ce2d8 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0xdc>
1003ce2fc:     	b	0x1003ce264 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x68>
1003ce300:     	mov	x19, x0
1003ce304:     	stp	x2, x1, [sp, #0x10]
1003ce308:     	mov	x1, x8
1003ce30c:     	mov	x2, x9
1003ce310:     	str	x3, [sp, #0x8]
1003ce314:     	mov	w3, #0x4                ; =4
1003ce318:     	mov	x28, x4
1003ce31c:     	mov	w4, #0x4                ; =4
1003ce320:     	bl	0x1028fab48 <__ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h181c4f681adaf166E>
1003ce324:     	mov	x0, x19
1003ce328:     	ldp	x2, x1, [sp, #0x10]
1003ce32c:     	mov	x4, x28
1003ce330:     	ldr	x3, [sp, #0x8]
1003ce334:     	ldr	x8, [x19, #0x10]
1003ce338:     	b	0x1003ce2bc <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0xc0>
1003ce33c:     	cmp	x22, x21
1003ce340:     	csel	x8, x22, x21, lo
1003ce344:     	cmp	x8, #0x4
1003ce348:     	b.hs	0x1003ce418 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x21c>
1003ce34c:     	mov	x8, #0x0                ; =0
1003ce350:     	b	0x1003ce438 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x23c>
1003ce354:     	sub	x21, x2, #0x1
1003ce358:     	mov	x22, x3
1003ce35c:     	b	0x1003ce37c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x180>
1003ce360:     	ldr	x8, [x0, #0x10]
1003ce364:     	str	x8, [x0, #0x10]
1003ce368:     	sub	x20, x20, #0x4
1003ce36c:     	add	x1, x1, #0x4
1003ce370:     	sub	x21, x21, #0x1
1003ce374:     	sub	x22, x22, #0x1
1003ce378:     	cbz	x22, 0x1003ce4a0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2a4>
1003ce37c:     	cmn	x21, #0x2
1003ce380:     	b.eq	0x1003ce4c0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2c4>
1003ce384:     	cbz	x20, 0x1003ce360 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x164>
1003ce388:     	udiv	x8, x21, x3
1003ce38c:     	add	x9, x8, #0x1
1003ce390:     	cmp	x4, x9
1003ce394:     	csinc	x9, x4, x8, lo
1003ce398:     	ldr	x8, [x0, #0x10]
1003ce39c:     	ldr	x10, [x0]
1003ce3a0:     	sub	x10, x10, x8
1003ce3a4:     	cmp	x9, x10
1003ce3a8:     	b.hi	0x1003ce3d4 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x1d8>
1003ce3ac:     	ldr	x9, [x0, #0x8]
1003ce3b0:     	ldr	s0, [x1], #0x4
1003ce3b4:     	str	s0, [x9, x8, lsl #2]
1003ce3b8:     	add	x8, x8, #0x1
1003ce3bc:     	str	x8, [x0, #0x10]
1003ce3c0:     	sub	x20, x20, #0x4
1003ce3c4:     	sub	x21, x21, #0x1
1003ce3c8:     	sub	x22, x22, #0x1
1003ce3cc:     	cbnz	x22, 0x1003ce37c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x180>
1003ce3d0:     	b	0x1003ce4a0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2a4>
1003ce3d4:     	mov	x19, x0
1003ce3d8:     	mov	x23, x1
1003ce3dc:     	mov	x1, x8
1003ce3e0:     	mov	x24, x2
1003ce3e4:     	mov	x2, x9
1003ce3e8:     	mov	x25, x3
1003ce3ec:     	mov	w3, #0x4                ; =4
1003ce3f0:     	mov	x26, x4
1003ce3f4:     	mov	w4, #0x4                ; =4
1003ce3f8:     	bl	0x1028fab48 <__ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h181c4f681adaf166E>
1003ce3fc:     	mov	x0, x19
1003ce400:     	mov	x1, x23
1003ce404:     	mov	x4, x26
1003ce408:     	mov	x3, x25
1003ce40c:     	mov	x2, x24
1003ce410:     	ldr	x8, [x19, #0x10]
1003ce414:     	b	0x1003ce3ac <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x1b0>
1003ce418:     	add	x8, x8, #0x1
1003ce41c:     	ands	x9, x8, #0x3
1003ce420:     	mov	w10, #0x4               ; =4
1003ce424:     	csel	x9, x10, x9, eq
1003ce428:     	sub	x8, x8, x9
1003ce42c:     	mov	x9, x8
1003ce430:     	subs	x9, x9, #0x4
1003ce434:     	b.ne	0x1003ce430 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x234>
1003ce438:     	sub	x9, x3, x8
1003ce43c:     	ands	x10, x9, #0x3
1003ce440:     	b.eq	0x1003ce458 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x25c>
1003ce444:     	cmp	x21, x8
1003ce448:     	b.eq	0x1003ce4c0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2c4>
1003ce44c:     	add	x8, x8, #0x1
1003ce450:     	subs	x10, x10, #0x1
1003ce454:     	b.ne	0x1003ce444 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x248>
1003ce458:     	sub	x9, x9, #0x1
1003ce45c:     	cmp	x9, #0x3
1003ce460:     	b.lo	0x1003ce4a0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2a4>
1003ce464:     	sub	x9, x2, x8
1003ce468:     	add	x9, x9, #0x1
1003ce46c:     	cbz	x9, 0x1003ce4c0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2c4>
1003ce470:     	cmp	x8, x2
1003ce474:     	b.eq	0x1003ce4c0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2c4>
1003ce478:     	add	x10, x8, #0x1
1003ce47c:     	cmp	x10, x2
1003ce480:     	b.eq	0x1003ce4c0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2c4>
1003ce484:     	add	x10, x8, #0x2
1003ce488:     	cmp	x10, x2
1003ce48c:     	b.eq	0x1003ce4c0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2c4>
1003ce490:     	add	x8, x8, #0x4
1003ce494:     	sub	x9, x9, #0x4
1003ce498:     	cmp	x8, x3
1003ce49c:     	b.ne	0x1003ce46c <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x270>
1003ce4a0:     	ldp	x29, x30, [sp, #0x70]
1003ce4a4:     	ldp	x20, x19, [sp, #0x60]
1003ce4a8:     	ldp	x22, x21, [sp, #0x50]
1003ce4ac:     	ldp	x24, x23, [sp, #0x40]
1003ce4b0:     	ldp	x26, x25, [sp, #0x30]
1003ce4b4:     	ldp	x28, x27, [sp, #0x20]
1003ce4b8:     	add	sp, sp, #0x80
1003ce4bc:     	ret
1003ce4c0:     	adrp	x3, 0x102dfc000 <_anon.e3f3e8898f3deed588652aef9995c150.21+0x2b0>
1003ce4c4:     	add	x3, x3, #0x348
1003ce4c8:     	add	x0, x2, #0x1
1003ce4cc:     	mov	x1, x2
1003ce4d0:     	bl	0x102982ca8 <__RNvNtNtCs6sq8b9ugfBC_4core5slice5index16slice_index_fail>
1003ce4d4:     	mov	x19, x0
1003ce4d8:     	mov	x21, x1
1003ce4dc:     	mov	x1, x8
1003ce4e0:     	mov	x20, x2
1003ce4e4:     	mov	x22, x3
1003ce4e8:     	mov	w3, #0x4                ; =4
1003ce4ec:     	mov	x23, x4
1003ce4f0:     	mov	w4, #0x4                ; =4
1003ce4f4:     	bl	0x1028fab48 <__ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h181c4f681adaf166E>
1003ce4f8:     	mov	x0, x19
1003ce4fc:     	mov	x1, x21
1003ce500:     	mov	x4, x23
1003ce504:     	mov	x3, x22
1003ce508:     	mov	x2, x20
1003ce50c:     	cbnz	x22, 0x1003ce234 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x38>
1003ce510:     	b	0x1003ce4a0 <__ZN19tenet_matrixalgebra9factorize5qr_lq24extend_adjoint_col_major17hd23af25dc88e1197E+0x2a4>
