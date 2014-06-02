class this.MMD
  constructor: (canvas, @width, @height) ->
    @gl = canvas.getContext('webgl') or canvas.getContext('experimental-webgl')
    if not @gl
      alert('WebGL not supported in your browser')
      throw 'WebGL not supported'

  initShaders: ->
    vshader = @gl.createShader(@gl.VERTEX_SHADER)
    @gl.shaderSource(vshader, MMD.VertexShaderSource)
    @gl.compileShader(vshader)
    if not @gl.getShaderParameter(vshader, @gl.COMPILE_STATUS)
      alert('Vertex shader compilation error')
      throw @gl.getShaderInfoLog(vshader)

    fshader = @gl.createShader(@gl.FRAGMENT_SHADER)
    @gl.shaderSource(fshader, MMD.FragmentShaderSource)
    @gl.compileShader(fshader)
    if not @gl.getShaderParameter(fshader, @gl.COMPILE_STATUS)
      alert('Fragment shader compilation error')
      throw @gl.getShaderInfoLog(fshader)

    @program = @gl.createProgram()
    @gl.attachShader(@program, vshader)
    @gl.attachShader(@program, fshader)

    @gl.linkProgram(@program)
    if not @gl.getProgramParameter(@program, @gl.LINK_STATUS)
      alert('Shader linking error')
      throw @gl.getProgramInfoLog(@program)

    @gl.useProgram(@program)

    attributes = []
    uniforms = []
    for src in [MMD.VertexShaderSource, MMD.FragmentShaderSource]
      for line in src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '').split(';')
        type = line.match(/^\s*(uniform|attribute)\s+/)?[1]
        continue if not type
        name = line.match(/(\w+)(\[\d+\])?\s*$/)[1]
        attributes.push(name) if type is 'attribute' and name not in attributes
        uniforms.push(name) if type is 'uniform' and name not in uniforms

    for name in attributes
      @program[name] = @gl.getAttribLocation(@program, name)
      @gl.enableVertexAttribArray(@program[name])

    for name in uniforms
      @program[name] = @gl.getUniformLocation(@program, name)

    return

  addModel: (model) ->
    @model = model # TODO: multi model?
    return

  initBuffers: ->
    @vbuffers = {}
    @initVertices()
    @initIndices()
    @initTextures()
    return

  initVertices: ->
    model = @model

    length = model.vertices.length
    weight = new Float32Array(length)
    vectors1 = new Float32Array(3 * length)
    vectors2 = new Float32Array(3 * length)
    rotations1 = new Float32Array(4 * length)
    rotations2 = new Float32Array(4 * length)
    positions1 = new Float32Array(3 * length)
    positions2 = new Float32Array(3 * length)
    morphVec = new Float32Array(3 * length)
    normals = new Float32Array(3 * length)
    uvs = new Float32Array(2 * length)
    edge = new Float32Array(length)
    for i in [0...length]
      vertex = model.vertices[i]
      bone1 = model.bones[vertex.bone_num1]
      bone2 = model.bones[vertex.bone_num2]
      weight[i] = vertex.bone_weight
      vectors1[3 * i    ] = vertex.x - bone1.head_pos[0]
      vectors1[3 * i + 1] = vertex.y - bone1.head_pos[1]
      vectors1[3 * i + 2] = vertex.z - bone1.head_pos[2]
      vectors2[3 * i    ] = vertex.x - bone2.head_pos[0]
      vectors2[3 * i + 1] = vertex.y - bone2.head_pos[1]
      vectors2[3 * i + 2] = vertex.z - bone2.head_pos[2]
      positions1[3 * i    ] = bone1.head_pos[0]
      positions1[3 * i + 1] = bone1.head_pos[1]
      positions1[3 * i + 2] = bone1.head_pos[2]
      positions2[3 * i    ] = bone2.head_pos[0]
      positions2[3 * i + 1] = bone2.head_pos[1]
      positions2[3 * i + 2] = bone2.head_pos[2]
      rotations1[4 * i + 3] = 1
      rotations2[4 * i + 3] = 1
      normals[3 * i    ] = vertex.nx
      normals[3 * i + 1] = vertex.ny
      normals[3 * i + 2] = vertex.nz
      uvs[2 * i    ] = vertex.u
      uvs[2 * i + 1] = vertex.v
      edge[i] = 1 - vertex.edge_flag
    model.rotations1 = rotations1
    model.rotations2 = rotations2
    model.positions1 = positions1
    model.positions2 = positions2
    model.morphVec = morphVec

    for data in [
      {attribute: 'aBoneWeight', array: weight, size: 1},
      {attribute: 'aVectorFromBone1', array: vectors1, size: 3},
      {attribute: 'aVectorFromBone2', array: vectors2, size: 3},
      {attribute: 'aBone1Rotation', array: rotations1, size: 4},
      {attribute: 'aBone2Rotation', array: rotations2, size: 4},
      {attribute: 'aBone1Position', array: positions1, size: 3},
      {attribute: 'aBone2Position', array: positions2, size: 3},
      {attribute: 'aMultiPurposeVector', array: morphVec, size: 3},
      {attribute: 'aVertexNormal', array: normals, size: 3},
      {attribute: 'aTextureCoord', array: uvs, size: 2},
      {attribute: 'aVertexEdge', array: edge, size: 1},
    ]
      buffer = @gl.createBuffer()
      @gl.bindBuffer(@gl.ARRAY_BUFFER, buffer)
      @gl.bufferData(@gl.ARRAY_BUFFER, data.array, @gl.STATIC_DRAW)
      @vbuffers[data.attribute] = {size: data.size, buffer: buffer}

    @gl.bindBuffer(@gl.ARRAY_BUFFER, null)
    return

  initIndices: ->
    indices = @model.triangles

    @ibuffer = @gl.createBuffer()
    @gl.bindBuffer(@gl.ELEMENT_ARRAY_BUFFER, @ibuffer)
    @gl.bufferData(@gl.ELEMENT_ARRAY_BUFFER, indices, @gl.STATIC_DRAW)
    @gl.bindBuffer(@gl.ELEMENT_ARRAY_BUFFER, null)
    return

  initTextures: ->
    model = @model

    @textureManager = new MMD.TextureManager(this)
    @textureManager.onload = => @redraw = true

    for material in model.materials
      material.textures = {} if not material.textures

      toonIndex = material.toon_index
      fileName = 'toon' + ('0' + (toonIndex + 1)).slice(-2) + '.bmp'
      if toonIndex == -1 or # -1 is special (no shadow)
        !model.toon_file_names or # no toon_file_names section in PMD
        fileName == model.toon_file_names[toonIndex] # toonXX.bmp is in 'data' directory
          fileName = 'lib/MMD.js/data/' + fileName
      else # otherwise the toon texture is in the model's directory
        fileName = model.directory + '/' + model.toon_file_names[toonIndex]
      material.textures.toon = @textureManager.get('toon', fileName)

      if material.texture_file_name
        for fileName in material.texture_file_name.split('*')
          switch fileName.slice(-4)
            when '.sph' then type = 'sph'
            when '.spa' then type = 'spa'
            when '.tga' then type = 'regular'; fileName += '.png'
            else             type = 'regular'
          material.textures[type] = @textureManager.get(type, model.directory + '/' + fileName)

    return

  start: ->
    @gl.clearColor(1, 1, 1, 1)
    @gl.clearDepth(1)
    @gl.enable(@gl.DEPTH_TEST)

    @redraw = true

    @shadowMap = new MMD.ShadowMap(this) if @drawSelfShadow
    @motionManager = new MMD.MotionManager

    count = 0
    t0 = before = Date.now()
    interval = 1000 / @fps

    step = =>
      @move()
      @computeMatrices()
      @render()

      now = Date.now()

      if ++count % @fps == 0
        @realFps = @fps / (now - before) * 1000
        before = now

      setTimeout(step, (t0 + count * interval) - now) # target_time - now

    step()
    return

  move: ->
    return if not @playing or @textureManager.pendingCount > 0
    if ++@frame > @motionManager.lastFrame
      @pause()
      @frame = -1
      return

    @moveCamera()
    @moveLight()
    @moveModel()
    return

  moveCamera: ->
    camera = @motionManager.getCameraFrame(@frame)
    if camera and not @ignoreCameraMotion
      @distance = camera.distance
      @rotx = camera.rotation[0]
      @roty = camera.rotation[1]
      @center = vec3.create(camera.location)
      @fovy = camera.view_angle

    return

  moveLight: ->
    light = @motionManager.getLightFrame(@frame)
    if light
      @lightDirection = light.location
      @lightColor = light.color

    return

  moveModel: ->
    model = @model
    {morphs, bones} = @motionManager.getModelFrame(model, @frame)

    @moveMorphs(model, morphs)
    @moveBones(model, bones)
    return

  moveMorphs: (model, morphs) ->
    return if not morphs
    return if model.morphs.length == 0

    for morph, j in model.morphs
      if j == 0
        base = morph
        continue
      continue if morph.name not of morphs
      weight = morphs[morph.name]
      for vert in morph.vert_data
        b = base.vert_data[vert.index]
        i = b.index
        model.morphVec[3 * i    ] += vert.x * weight
        model.morphVec[3 * i + 1] += vert.y * weight
        model.morphVec[3 * i + 2] += vert.z * weight

    @gl.bindBuffer(@gl.ARRAY_BUFFER, @vbuffers.aMultiPurposeVector.buffer)
    @gl.bufferData(@gl.ARRAY_BUFFER, model.morphVec, @gl.STATIC_DRAW)
    @gl.bindBuffer(@gl.ARRAY_BUFFER, null)

    # reset positions
    for b in base.vert_data
      i = b.index
      model.morphVec[3 * i    ] = 0
      model.morphVec[3 * i + 1] = 0
      model.morphVec[3 * i + 2] = 0

    return

  moveBones: (model, bones) ->
    return if not bones

    # individualBoneMotions is translation/rotation of each bone from it's original position
    # boneMotions is total position/rotation of each bone
    # boneMotions is an array like [{p, r, tainted}]
    # tainted flag is used to avoid re-creating vec3/quat4
    individualBoneMotions = []
    boneMotions = []
    originalBonePositions = []
    parentBones = []
    constrainedBones = []

    for bone, i in model.bones
      individualBoneMotions[i] = bones[bone.name] ? {
        rotation: quat4.create([0, 0, 0, 1])
        location: vec3.create()
      }
      boneMotions[i] = {
        r: quat4.create()
        p: vec3.create()
        tainted: true
      }
      originalBonePositions[i] = bone.head_pos
      parentBones[i] = bone.parent_bone_index
      if bone.name.indexOf('\u3072\u3056') > 0 # $B$R$6(B
        constrainedBones[i] = true # TODO: for now it's only for knees, but extend this if I do PMX

    getBoneMotion = (boneIndex) ->
      # http://d.hatena.ne.jp/edvakf/20111026/1319656727
      motion = boneMotions[boneIndex]
      return motion if motion and not motion.tainted

      m = individualBoneMotions[boneIndex]
      r = quat4.set(m.rotation, motion.r)
      t = m.location
      p = vec3.set(originalBonePositions[boneIndex], motion.p)

      if parentBones[boneIndex] == 0xFFFF # center, foot IK, etc.
        return boneMotions[boneIndex] = {
          p: vec3.add(p, t),
          r: r
          tainted: false
        }
      else
        parentIndex = parentBones[boneIndex]
        parentMotion = getBoneMotion(parentIndex)
        r = quat4.multiply(parentMotion.r, r, r)
        p = vec3.subtract(p, originalBonePositions[parentIndex])
        vec3.add(p, t)
        vec3.rotateByQuat4(p, parentMotion.r)
        vec3.add(p, parentMotion.p)
        return boneMotions[boneIndex] = {p: p, r: r, tainted: false}

    resolveIKs = ->
      # this function is run only once, but to narrow the scope I'm making a function
      # http://d.hatena.ne.jp/edvakf/20111102/1320268602

      # objects to be reused
      targetVec = vec3.create()
      ikboneVec = vec3.create()
      axis = vec3.create()
      tmpQ = quat4.create()
      tmpR = quat4.create()

      for ik in model.iks
        ikbonePos = getBoneMotion(ik.bone_index).p
        targetIndex = ik.target_bone_index
        minLength = 0.1 * vec3.length(
          vec3.subtract(
            originalBonePositions[targetIndex],
            originalBonePositions[parentBones[targetIndex]], axis)) # temporary use of axis

        for n in [0...ik.iterations]
          targetPos = getBoneMotion(targetIndex).p # this should calculate the whole chain
          break if minLength > vec3.length(
            vec3.subtract(targetPos, ikbonePos, axis)) # temporary use of axis

          for boneIndex, i in ik.child_bones
            motion = getBoneMotion(boneIndex)
            bonePos = motion.p
            targetPos = getBoneMotion(targetIndex).p if i > 0
            targetVec = vec3.subtract(targetPos, bonePos, targetVec)
            targetVecLen = vec3.length(targetVec)
            continue if targetVecLen < minLength # targetPos == bonePos
            ikboneVec = vec3.subtract(ikbonePos, bonePos, ikboneVec)
            ikboneVecLen = vec3.length(ikboneVec)
            continue if ikboneVecLen < minLength # ikbonePos == bonePos
            axis = vec3.cross(targetVec, ikboneVec, axis)
            axisLen = vec3.length(axis)
            sinTheta = axisLen / ikboneVecLen / targetVecLen
            continue if sinTheta < 0.001 # ~0.05 degree
            maxangle = (i + 1) * ik.control_weight * 4 # angle to move in one iteration
            theta = Math.asin(sinTheta)
            theta = 3.141592653589793 - theta if vec3.dot(targetVec, ikboneVec) < 0
            theta = maxangle if theta > maxangle
            q = quat4.set(vec3.scale(axis, Math.sin(theta / 2) / axisLen), tmpQ) # q is tmpQ
            q[3] = Math.cos(theta / 2)
            parentRotation = getBoneMotion(parentBones[boneIndex]).r
            r = quat4.inverse(parentRotation, tmpR) # r is tmpR
            r = quat4.multiply(quat4.multiply(r, q), motion.r)

            if constrainedBones[boneIndex]
              c = r[3] # cos(theta / 2)
              r = quat4.set([Math.sqrt(1 - c * c), 0, 0, c], r) # axis must be x direction
              quat4.inverse(boneMotions[boneIndex].r, q)
              quat4.multiply(r, q, q)
              q = quat4.multiply(parentRotation, q, q)

            # update individualBoneMotions[boneIndex].rotation
            quat4.normalize(r, individualBoneMotions[boneIndex].rotation)
            # update boneMotions[boneIndex].r which is the same as motion.r
            quat4.multiply(q, motion.r, motion.r)

            # taint for re-calculation
            boneMotions[ik.child_bones[j]].tainted = true for j in [0...i]
            boneMotions[ik.target_bone_index].tainted = true

    resolveIKs()

    # calculate positions/rotations of bones other than IK
    getBoneMotion(i) for i in [0...model.bones.length]

    #TODO: split

    rotations1 = model.rotations1
    rotations2 = model.rotations2
    positions1 = model.positions1
    positions2 = model.positions2

    length = model.vertices.length
    for i in [0...length]
      vertex = model.vertices[i]
      motion1 = boneMotions[vertex.bone_num1]
      motion2 = boneMotions[vertex.bone_num2]
      rot1 = motion1.r
      pos1 = motion1.p
      rot2 = motion2.r
      pos2 = motion2.p
      rotations1[i * 4    ] = rot1[0]
      rotations1[i * 4 + 1] = rot1[1]
      rotations1[i * 4 + 2] = rot1[2]
      rotations1[i * 4 + 3] = rot1[3]
      rotations2[i * 4    ] = rot2[0]
      rotations2[i * 4 + 1] = rot2[1]
      rotations2[i * 4 + 2] = rot2[2]
      rotations2[i * 4 + 3] = rot2[3]
      positions1[i * 3    ] = pos1[0]
      positions1[i * 3 + 1] = pos1[1]
      positions1[i * 3 + 2] = pos1[2]
      positions2[i * 3    ] = pos2[0]
      positions2[i * 3 + 1] = pos2[1]
      positions2[i * 3 + 2] = pos2[2]

    @gl.bindBuffer(@gl.ARRAY_BUFFER, @vbuffers.aBone1Rotation.buffer)
    @gl.bufferData(@gl.ARRAY_BUFFER, rotations1, @gl.STATIC_DRAW)
    @gl.bindBuffer(@gl.ARRAY_BUFFER, @vbuffers.aBone2Rotation.buffer)
    @gl.bufferData(@gl.ARRAY_BUFFER, rotations2, @gl.STATIC_DRAW)
    @gl.bindBuffer(@gl.ARRAY_BUFFER, @vbuffers.aBone1Position.buffer)
    @gl.bufferData(@gl.ARRAY_BUFFER, positions1, @gl.STATIC_DRAW)
    @gl.bindBuffer(@gl.ARRAY_BUFFER, @vbuffers.aBone2Position.buffer)
    @gl.bufferData(@gl.ARRAY_BUFFER, positions2, @gl.STATIC_DRAW)
    @gl.bindBuffer(@gl.ARRAY_BUFFER, null)
    return

  computeMatrices: ->
    @modelMatrix = mat4.createIdentity() # model aligned with the world for now

    @cameraPosition = vec3.create([0, 0, @distance]) # camera position in world space
    vec3.rotateX(@cameraPosition, @rotx)
    vec3.rotateY(@cameraPosition, @roty)
    vec3.moveBy(@cameraPosition, @center)

    up = [0, 1, 0]
    vec3.rotateX(up, @rotx)
    vec3.rotateY(up, @roty)

    @viewMatrix = mat4.lookAt(@cameraPosition, @center, up)

    @mvMatrix = mat4.createMultiply(@viewMatrix, @modelMatrix)

    @pMatrix = mat4.perspective(@fovy, @width / @height, 0.1, 1000.0)

    # normal matrix; inverse transpose of mvMatrix
    # model -> view space; only applied to directional vectors (not points)
    @nMatrix = mat4.inverseTranspose(@mvMatrix, mat4.create())
    return

  render: ->
    return if not @redraw and not @playing
    @redraw = false

    @gl.bindFramebuffer(@gl.FRAMEBUFFER, null)
    @gl.viewport(0, 0, @width, @height)
    @gl.clear(@gl.COLOR_BUFFER_BIT | @gl.DEPTH_BUFFER_BIT)

    for attribute, vb of @vbuffers
      @gl.bindBuffer(@gl.ARRAY_BUFFER, vb.buffer)
      @gl.vertexAttribPointer(@program[attribute], vb.size, @gl.FLOAT, false, 0, 0)

    @gl.bindBuffer(@gl.ELEMENT_ARRAY_BUFFER, @ibuffer)

    @setSelfShadowTexture()

    @setUniforms()

    @gl.enable(@gl.CULL_FACE)
    @gl.enable(@gl.BLEND)
    @gl.blendFuncSeparate(@gl.SRC_ALPHA, @gl.ONE_MINUS_SRC_ALPHA, @gl.SRC_ALPHA, @gl.DST_ALPHA)

    offset = 0
    for material in @model.materials
      @renderMaterial(material, offset)
      offset += material.face_vert_count

    @gl.disable(@gl.BLEND)

    offset = 0
    for material in @model.materials
      @renderEdge(material, offset)
      offset += material.face_vert_count

    @gl.disable(@gl.CULL_FACE)

    @renderAxes()

    @gl.flush()
    return

  setSelfShadowTexture: ->
    return if not @drawSelfShadow

    model = @model

    @shadowMap.computeMatrices()
    @shadowMap.beforeRender()

    offset = 0
    for material in model.materials
      continue if 0.979 < material.alpha < 0.981 # alpha is 0.98

      @gl.drawElements(@gl.TRIANGLES, material.face_vert_count, @gl.UNSIGNED_SHORT, offset * 2)
      offset += material.face_vert_count

    @shadowMap.afterRender()

    @gl.activeTexture(@gl.TEXTURE3) # 3 -> shadow map
    @gl.bindTexture(@gl.TEXTURE_2D, @shadowMap.getTexture())
    @gl.uniform1i(@program.uShadowMap, 3)
    @gl.uniformMatrix4fv(@program.uLightMatrix, false, @shadowMap.getLightMatrix())
    @gl.uniform1i(@program.uSelfShadow, true)

    # reset
    @gl.bindFramebuffer(@gl.FRAMEBUFFER, null)
    @gl.viewport(0, 0, @width, @height) # not needed on Windows Chrome but necessary on Mac Chrome
    return

  setUniforms: ->
    @gl.uniform1f(@program.uEdgeThickness, @edgeThickness)
    @gl.uniform3fv(@program.uEdgeColor, @edgeColor)
    @gl.uniformMatrix4fv(@program.uMVMatrix, false, @mvMatrix)
    @gl.uniformMatrix4fv(@program.uPMatrix, false, @pMatrix)
    @gl.uniformMatrix4fv(@program.uNMatrix, false, @nMatrix)

    # direction of light source defined in world space, then transformed to view space
    lightDirection = vec3.createNormalize(@lightDirection) # world space
    mat4.multiplyVec3(@nMatrix, lightDirection) # view space
    @gl.uniform3fv(@program.uLightDirection, lightDirection)

    @gl.uniform3fv(@program.uLightColor, @lightColor)
    return

  renderMaterial: (material, offset) ->
    @gl.uniform3fv(@program.uAmbientColor, material.ambient)
    @gl.uniform3fv(@program.uSpecularColor, material.specular)
    @gl.uniform3fv(@program.uDiffuseColor, material.diffuse)
    @gl.uniform1f(@program.uAlpha, material.alpha)
    @gl.uniform1f(@program.uShininess, material.shininess)
    @gl.uniform1i(@program.uEdge, false)

    textures = material.textures

    @gl.activeTexture(@gl.TEXTURE0) # 0 -> toon
    @gl.bindTexture(@gl.TEXTURE_2D, textures.toon)
    @gl.uniform1i(@program.uToon, 0)

    if textures.regular
      @gl.activeTexture(@gl.TEXTURE1) # 1 -> regular texture
      @gl.bindTexture(@gl.TEXTURE_2D, textures.regular)
      @gl.uniform1i(@program.uTexture, 1)
    @gl.uniform1i(@program.uUseTexture, !!textures.regular)

    if textures.sph or textures.spa
      @gl.activeTexture(@gl.TEXTURE2) # 2 -> sphere map texture
      @gl.bindTexture(@gl.TEXTURE_2D, textures.sph || textures.spa)
      @gl.uniform1i(@program.uSphereMap, 2)
      @gl.uniform1i(@program.uUseSphereMap, true)
      @gl.uniform1i(@program.uIsSphereMapAdditive, !!textures.spa)
    else
      @gl.uniform1i(@program.uUseSphereMap, false)

    @gl.cullFace(@gl.BACK)

    @gl.drawElements(@gl.TRIANGLES, material.face_vert_count, @gl.UNSIGNED_SHORT, offset * 2)

    return

  renderEdge: (material, offset) ->
    return if not @drawEdge or not material.edge_flag

    @gl.uniform1i(@program.uEdge, true)
    @gl.cullFace(@gl.FRONT)

    @gl.drawElements(@gl.TRIANGLES, material.face_vert_count, @gl.UNSIGNED_SHORT, offset * 2)

    @gl.cullFace(@gl.BACK)
    @gl.uniform1i(@program.uEdge, false)

  renderAxes: ->

    axisBuffer = @gl.createBuffer()
    @gl.bindBuffer(@gl.ARRAY_BUFFER, axisBuffer)
    @gl.vertexAttribPointer(@program.aMultiPurposeVector, 3, @gl.FLOAT, false, 0, 0)
    if @drawAxes
      @gl.uniform1i(@program.uAxis, true)

      for i in [0...3]
        axis = [0, 0, 0, 0, 0, 0]
        axis[i] = 65 # from [65, 0, 0] to [0, 0, 0] etc.
        color = [0, 0, 0]
        color[i] = 1
        @gl.bufferData(@gl.ARRAY_BUFFER, new Float32Array(axis), @gl.STATIC_DRAW)
        @gl.uniform3fv(@program.uAxisColor, color)
        @gl.drawArrays(@gl.LINES, 0, 2)

      axis = [
        -50, 0, 0, 0, 0, 0 # negative x-axis (from [-50, 0, 0] to origin)
        0, 0, -50, 0, 0, 0 # negative z-axis (from [0, 0, -50] to origin)
      ]
      for i in [-50..50] by 5
        if i != 0
          axis.push(
            i,   0, -50,
            i,   0, 50, # one line parallel to the x-axis
            -50, 0, i,
            50,  0, i   # one line parallel to the z-axis
          )
      color = [0.7, 0.7, 0.7]
      @gl.bufferData(@gl.ARRAY_BUFFER, new Float32Array(axis), @gl.STATIC_DRAW)
      @gl.uniform3fv(@program.uAxisColor, color)
      @gl.drawArrays(@gl.LINES, 0, 84)

      @gl.uniform1i(@program.uAxis, false)

    # draw center point
    if @drawCenterPoint
      @gl.uniform1i(@program.uCenterPoint, true)
      @gl.bufferData(@gl.ARRAY_BUFFER, new Float32Array(@center), @gl.STATIC_DRAW)
      @gl.drawArrays(@gl.POINTS, 0, 1)
      @gl.uniform1i(@program.uCenterPoint, false)

    @gl.deleteBuffer(axisBuffer)
    @gl.bindBuffer(@gl.ARRAY_BUFFER, null)
    return

  registerKeyListener: (element) ->
    element.addEventListener('keydown', (e) =>
      switch e.keyCode + e.shiftKey * 1000 + e.ctrlKey * 10000 + e.altKey * 100000
        when 37 then @roty += Math.PI / 12 # left
        when 39 then @roty -= Math.PI / 12 # right
        when 38 then @rotx += Math.PI / 12 # up
        when 40 then @rotx -= Math.PI / 12 # down
        when 33 then @distance -= 3 * @distance / @DIST # pageup
        when 34 then @distance += 3 * @distance / @DIST # pagedown
        when 36 # home
          @rotx = @roty = 0
          @center = [0, 10, 0]
          @distance = @DIST
        when 1037 # shift + left
          vec3.multiplyMat4(@center, @mvMatrix)
          @center[0] -= @distance / @DIST
          vec3.multiplyMat4(@center, mat4.createInverse(@mvMatrix))
        when 1039 # shift + right
          vec3.multiplyMat4(@center, @mvMatrix)
          @center[0] += @distance / @DIST
          vec3.multiplyMat4(@center, mat4.createInverse(@mvMatrix))
        when 1038 # shift +  up
          vec3.multiplyMat4(@center, @mvMatrix)
          @center[1] += @distance / @DIST
          vec3.multiplyMat4(@center, mat4.createInverse(@mvMatrix))
        when 1040 # shift + down
          vec3.multiplyMat4(@center, @mvMatrix)
          @center[1] -= @distance / @DIST
          vec3.multiplyMat4(@center, mat4.createInverse(@mvMatrix))
        when 32 # space
          if @playing
            @pause()
          else
            @play()
        else return

      e.preventDefault()
      @redraw = true
    , false)
    return

  registerMouseListener: (element) ->
    @registerDragListener(element)
    @registerWheelListener(element)
    return

  registerDragListener: (element) ->
    element.addEventListener('mousedown', (e) =>
      return if e.button != 0
      modifier = e.shiftKey * 1000 + e.ctrlKey * 10000 + e.altKey * 100000
      return if modifier != 0 and modifier != 1000
      ox = e.clientX; oy = e.clientY

      move = (dx, dy, modi) =>
        if modi == 0
          @roty -= dx / 100
          @rotx -= dy / 100
          @redraw = true
        else if modi == 1000
          vec3.multiplyMat4(@center, @mvMatrix)
          @center[0] -= dx / 30 * @distance / @DIST
          @center[1] += dy / 30 * @distance / @DIST
          vec3.multiplyMat4(@center, mat4.createInverse(@mvMatrix))
          @redraw = true

      onmouseup = (e) =>
        return if e.button != 0
        modi = e.shiftKey * 1000 + e.ctrlKey * 10000 + e.altKey * 100000
        move(e.clientX - ox, e.clientY - oy, modi)
        element.removeEventListener('mouseup', onmouseup, false)
        element.removeEventListener('mousemove', onmousemove, false)
        e.preventDefault()

      onmousemove = (e) =>
        return if e.button != 0
        modi = e.shiftKey * 1000 + e.ctrlKey * 10000 + e.altKey * 100000
        x = e.clientX; y = e.clientY
        move(x - ox, y - oy, modi)
        ox = x; oy = y
        e.preventDefault()

      element.addEventListener('mouseup', onmouseup, false)
      element.addEventListener('mousemove', onmousemove, false)
    , false)
    return

  registerWheelListener: (element) ->
    onwheel = (e) =>
      delta = e.detail || e.wheelDelta / (-40) # positive: wheel down
      @distance += delta * @distance / @DIST
      @redraw = true
      e.preventDefault()

    if 'onmousewheel' of window
      element.addEventListener('mousewheel', onwheel, false)
    else
      element.addEventListener('DOMMouseScroll', onwheel, false)

    return

  initParameters: ->
    # camera/view settings
    @ignoreCameraMotion = false
    @rotx = @roty = 0
    @distance = @DIST = 35
    @center = [0, 10, 0]
    @fovy = 40

    # edge
    @drawEdge = true
    @edgeThickness = 0.004
    @edgeColor = [0, 0, 0]

    # light
    @lightDirection = [0.5, 1.0, 0.5]
    @lightDistance = 8875
    @lightColor = [0.6, 0.6, 0.6]

    # misc
    @drawSelfShadow = true
    @drawAxes = true
    @drawCenterPoint = false

    @fps = 30 # redraw every 1000/30 msec
    @realFps = @fps
    @playing = false
    @frame = -1
    return

  addCameraLightMotion: (motion, merge_flag, frame_offset) ->
    @motionManager.addCameraLightMotion(motion, merge_flag, frame_offset)
    return

  addModelMotion: (model, motion, merge_flag, frame_offset) ->
    @motionManager.addModelMotion(model, motion, merge_flag, frame_offset)
    return

  play: ->
    @playing = true
    return

  pause: ->
    @playing = false
    return

  rewind: ->
    @setFrameNumber(-1)
    return

  setFrameNumber: (num) ->
    @frame = num
    return


MMD.FragmentShaderSource = '''

  #ifdef GL_ES
  precision highp float;
  #endif

  varying vec2 vTextureCoord;
  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec4 vLightCoord;

  uniform vec3 uLightDirection; // light source direction in world space
  uniform vec3 uLightColor;

  uniform vec3 uAmbientColor;
  uniform vec3 uSpecularColor;
  uniform vec3 uDiffuseColor;
  uniform float uAlpha;
  uniform float uShininess;

  uniform bool uUseTexture;
  uniform bool uUseSphereMap;
  uniform bool uIsSphereMapAdditive;

  uniform sampler2D uToon;
  uniform sampler2D uTexture;
  uniform sampler2D uSphereMap;

  uniform bool uEdge;
  uniform float uEdgeThickness;
  uniform vec3 uEdgeColor;

  uniform bool uGenerateShadowMap;
  uniform bool uSelfShadow;
  uniform sampler2D uShadowMap;

  uniform bool uAxis;
  uniform vec3 uAxisColor;
  uniform bool uCenterPoint;

  // from http://spidergl.org/example.php?id=6
  vec4 pack_depth(const in float depth) {
    const vec4 bit_shift = vec4(256.0*256.0*256.0, 256.0*256.0, 256.0, 1.0);
    const vec4 bit_mask  = vec4(0.0, 1.0/256.0, 1.0/256.0, 1.0/256.0);
    vec4 res = fract(depth * bit_shift);
    res -= res.xxyz * bit_mask;
    return res;
  }
  float unpack_depth(const in vec4 rgba_depth)
  {
    const vec4 bit_shift = vec4(1.0/(256.0*256.0*256.0), 1.0/(256.0*256.0), 1.0/256.0, 1.0);
    float depth = dot(rgba_depth, bit_shift);
    return depth;
  }

  void main() {
    if (uGenerateShadowMap) {
      //gl_FragData[0] = pack_depth(gl_FragCoord.z);
      gl_FragColor = pack_depth(gl_FragCoord.z);
      return;
    }
    if (uAxis) {
      gl_FragColor = vec4(uAxisColor, 1.0);
      return;
    }
    if (uCenterPoint) {
      vec2 uv = gl_PointCoord * 2.0 - 1.0; // transform [0, 1] -> [-1, 1] coord systems
      float w = dot(uv, uv);
      if (w < 0.3 || (w > 0.5 && w < 1.0)) {
        gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
      } else {
        discard;
      }
      return;
    }

    // vectors are in view space
    vec3 norm = normalize(vNormal); // each point's normal vector in view space
    vec3 cameraDirection = normalize(-vPosition); // camera located at origin in view space

    vec3 color;
    float alpha = uAlpha;

    if (uEdge) {

      color = uEdgeColor;

    } else {

      color = vec3(1.0, 1.0, 1.0);
      if (uUseTexture) {
        vec4 texColor = texture2D(uTexture, vTextureCoord);
        color *= texColor.rgb;
        alpha *= texColor.a;
      }
      if (uUseSphereMap) {
        vec2 sphereCoord = 0.5 * (1.0 + vec2(1.0, -1.0) * norm.xy);
        if (uIsSphereMapAdditive) {
          color += texture2D(uSphereMap, sphereCoord).rgb;
        } else {
          color *= texture2D(uSphereMap, sphereCoord).rgb;
        }
      }

      // specular component
      vec3 halfAngle = normalize(uLightDirection + cameraDirection);
      float specularWeight = pow( max(0.001, dot(halfAngle, norm)) , uShininess );
      //float specularWeight = pow( max(0.0, dot(reflect(-uLightDirection, norm), cameraDirection)) , uShininess ); // another definition
      vec3 specular = specularWeight * uSpecularColor;

      vec2 toonCoord = vec2(0.0, 0.5 * (1.0 - dot( uLightDirection, norm )));

      if (uSelfShadow) {
        vec3 lightCoord = vLightCoord.xyz / vLightCoord.w; // projection to texture coordinate (in light space)
        vec4 rgbaDepth = texture2D(uShadowMap, lightCoord.xy);
        float depth = unpack_depth(rgbaDepth);
        if (depth < lightCoord.z - 0.01) {
          toonCoord = vec2(0.0, 0.55);
        }
      }

      color *= uAmbientColor + uLightColor * (uDiffuseColor + specular);

      color = clamp(color, 0.0, 1.0);
      color *= texture2D(uToon, toonCoord).rgb;

    }
    gl_FragColor = vec4(color, alpha);

  }

'''

# see http://blog.goo.ne.jp/torisu_tetosuki/e/209ad341d3ece2b1b4df24abf619d6e4

# some shorthands
size_Int8 = Int8Array.BYTES_PER_ELEMENT
size_Uint8 = Uint8Array.BYTES_PER_ELEMENT
size_Uint16 = Uint16Array.BYTES_PER_ELEMENT
size_Uint32 = Uint32Array.BYTES_PER_ELEMENT
size_Float32 = Float32Array.BYTES_PER_ELEMENT

slice = Array.prototype.slice

class this.MMD.Model # export to top level
  constructor: (directory, filename) ->
    @directory = directory
    @filename = filename
    @vertices = null
    @triangles = null
    @materials = null
    @bones = null
    @morphs = null
    @morph_order = null
    @bone_group_names = null
    @bone_table = null
    @english_flag = null
    @english_name = null
    @english_comment = null
    @english_bone_names = null
    @english_morph_names = null
    @english_bone_group_names = null
    @toon_file_names = null
    @rigid_bodies = null
    @joints = null

  load: (callback) ->
    xhr = new XMLHttpRequest
    xhr.open('GET', @directory + '/' + @filename, true)
    xhr.responseType = 'arraybuffer'
    xhr.onload = =>
      console.time('parse')
      @parse(xhr.response)
      console.timeEnd('parse')
      callback()
    xhr.send()

  parse: (buffer) ->
    length = buffer.byteLength
    view = new DataView(buffer, 0)
    offset = 0
    offset = @checkHeader(buffer, view, offset)
    offset = @getName(buffer, view, offset)
    offset = @getVertices(buffer, view, offset)
    offset = @getTriangles(buffer, view, offset)
    offset = @getMaterials(buffer, view, offset)
    offset = @getBones(buffer, view, offset)
    offset = @getIKs(buffer, view, offset)
    offset = @getMorphs(buffer, view, offset)
    offset = @getMorphOrder(buffer, view, offset)
    offset = @getBoneGroupNames(buffer, view, offset)
    offset = @getBoneTable(buffer, view, offset)
    return if (offset >= length)
    offset = @getEnglishFlag(buffer, view, offset)
    if @english_flag
      offset = @getEnglishName(buffer, view, offset)
      offset = @getEnglishBoneNames(buffer, view, offset)
      offset = @getEnglishMorphNames(buffer, view, offset)
      offset = @getEnglishBoneGroupNames(buffer, view, offset)
    return if (offset >= length)
    offset = @getToonFileNames(buffer, view, offset)
    return if (offset >= length)
    offset = @getRigidBodies(buffer, view, offset)
    offset = @getJoints(buffer, view, offset)

  checkHeader: (buffer, view, offset) ->
    if view.getUint8(0) != 'P'.charCodeAt(0) or
       view.getUint8(1) != 'm'.charCodeAt(0) or
       view.getUint8(2) != 'd'.charCodeAt(0) or
       view.getUint8(3) != 0x00 or
       view.getUint8(4) != 0x00 or
       view.getUint8(5) != 0x80 or
       view.getUint8(6) != 0x3F
      throw 'File is not PMD'
    offset += 7 * size_Uint8

  getName: (buffer, view, offset) ->
    block = new Uint8Array(buffer, offset, 20 + 256)
    @name = sjisArrayToString(slice.call(block, 0, 20))
    @comment = sjisArrayToString(slice.call(block, 20, 20 + 256))
    offset += (20 + 256) * size_Uint8

  getVertices: (buffer, view, offset) ->
    length = view.getUint32(offset, true)
    offset += size_Uint32
    @vertices =
      for i in [0...length]
        new Vertex(buffer, view, offset + i * Vertex.size)
    offset += length * Vertex.size

  getTriangles: (buffer, view, offset) ->
    length = view.getUint32(offset, true)
    offset += size_Uint32
    @triangles = new Uint16Array(length)
    #left->right handed system (swap 0th and 1st vertices)
    for i in [0...length] by 3
      @triangles[i + 1] = view.getUint16(offset + i * size_Uint16, true)
      @triangles[i] = view.getUint16(offset + (i + 1) * size_Uint16, true)
      @triangles[i + 2] = view.getUint16(offset + (i + 2) * size_Uint16, true)
    offset += length * size_Uint16

  getMaterials: (buffer, view, offset) ->
    length = view.getUint32(offset, true)
    offset += size_Uint32
    @materials =
      for i in [0...length]
        new Material(buffer, view, offset + i * Material.size)
    offset += length * Material.size

  getBones: (buffer, view, offset) ->
    length = view.getUint16(offset, true)
    offset += size_Uint16
    @bones =
      for i in [0...length]
        new Bone(buffer, view, offset + i * Bone.size)
    offset += length * Bone.size

  getIKs: (buffer, view, offset) ->
    length = view.getUint16(offset, true)
    offset += size_Uint16
    @iks =
      for i in [0...length]
        ik = new IK(buffer, view, offset)
        offset += ik.getSize()
        ik
    offset

  getMorphs: (buffer, view, offset) ->
    length = view.getUint16(offset, true)
    offset += size_Uint16
    @morphs =
      for i in [0...length]
        morph = new Morph(buffer, view, offset)
        offset += morph.getSize()
        morph
    offset

  getMorphOrder: (buffer, view, offset) ->
    length = view.getUint8(offset)
    offset += size_Uint8
    @morph_order =
      for i in [0...length]
        view.getUint16(offset + i * size_Uint16, true)
    offset += length * size_Uint16

  getBoneGroupNames: (buffer, view, offset) ->
    length = view.getUint8(offset)
    offset += size_Uint8
    block = new Uint8Array(buffer, offset, 50 * length)
    @bone_group_names =
      for i in [0...length]
        sjisArrayToString(slice.call(block, i * 50, (i + 1) * 50))
    offset += length * 50 * size_Uint8

  getBoneTable: (buffer, view, offset) ->
    length = view.getUint32(offset, true)
    offset += size_Uint32
    @bone_table =
      for i in [0...length]
        bone = {}
        bone.index = view.getUint16(offset, true); offset += size_Uint16
        bone.group_index = view.getUint8(offset); offset += size_Uint8
        bone
    offset

  getEnglishFlag: (buffer, view, offset) ->
    @english_flag = view.getUint8(offset)
    offset += size_Uint8

  getEnglishName: (buffer, view, offset) ->
    block = new Uint8Array(buffer, offset, 20 + 256)
    @english_name = sjisArrayToString(slice.call(block, 0, 20))
    @english_comment = sjisArrayToString(slice.call(block, 20, 20 + 256))
    offset += (20 + 256) * size_Uint8

  getEnglishBoneNames: (buffer, view, offset) ->
    length = @bones.length
    block = new Uint8Array(buffer, offset, 20 * length)
    @english_bone_names =
      for i in [0...length]
        sjisArrayToString(slice.call(block, i * 20, (i + 1) * 20))
    offset += length * 20 * size_Uint8

  getEnglishMorphNames: (buffer, view, offset) ->
    length = @morphs.length - 1
    length = 0 if length < 0
    block = new Uint8Array(buffer, offset, 20 * length)
    @english_morph_names =
      for i in [0...length]
        sjisArrayToString(slice.call(block, i * 20, (i + 1) * 20))
    offset += length * 20 * size_Uint8

  getEnglishBoneGroupNames: (buffer, view, offset) ->
    length = @bone_group_names.length
    block = new Uint8Array(buffer, offset, 50 * length)
    @english_bone_group_names =
      for i in [0...length]
        sjisArrayToString(slice.call(block, i * 50, (i + 1) * 50))
    offset += length * 50 * size_Uint8

  getToonFileNames: (buffer, view, offset) ->
    block = new Uint8Array(buffer, offset, 100 * 10)
    @toon_file_names =
      for i in [0...10]
        sjisArrayToString(slice.call(block, i * 100, (i + 1) * 100))
    offset += 100 * 10 * size_Uint8

  getRigidBodies: (buffer, view, offset) ->
    length = view.getUint32(offset, true)
    offset += size_Uint32
    @rigid_bodies =
      for i in [0...length]
        new RigidBody(buffer, view, offset + i * RigidBody.size)
    offset += length * RigidBody.size

  getJoints: (buffer, view, offset) ->
    length = view.getUint32(offset, true)
    offset += size_Uint32
    @joints =
      for i in [0...length]
        new Joint(buffer, view, offset + i * Joint.size)
    offset += length * Joint.size

#http://blog.goo.ne.jp/torisu_tetosuki/e/5a1b16e2f61067838dfc66d010389707
#float pos[3]; // x, y, z // 座標
#float normal_vec[3]; // nx, ny, nz // 法線ベクトル
#float uv[2]; // u, v // UV座標 // MMDは頂点UV
#WORD bone_num[2]; // ボーン番号1、番号2 // モデル変形(頂点移動)時に影響
#BYTE bone_weight; // ボーン1に与える影響度 // min:0 max:100 // ボーン2への影響度は、(100 - bone_weight)
#BYTE edge_flag; // 0:通常、1:エッジ無効 // エッジ(輪郭)が有効の場合
class Vertex
  constructor: (buffer, view, offset) ->
    @x = view.getFloat32(offset, true); offset += size_Float32
    @y = view.getFloat32(offset, true); offset += size_Float32
    @z = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    @nx = view.getFloat32(offset, true); offset += size_Float32
    @ny = view.getFloat32(offset, true); offset += size_Float32
    @nz = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    @u = view.getFloat32(offset, true); offset += size_Float32
    @v = view.getFloat32(offset, true); offset += size_Float32
    @bone_num1 = view.getUint16(offset, true); offset += size_Uint16
    @bone_num2 = view.getUint16(offset, true); offset += size_Uint16
    @bone_weight = view.getUint8(offset); offset += size_Uint8
    @edge_flag = view.getUint8(offset); offset += size_Uint8

Vertex.size = size_Float32 * 8 + size_Uint16 * 2 + size_Uint8 * 2 # 38

#http://blog.goo.ne.jp/torisu_tetosuki/e/ea0bb1b1d4c6ad98a93edbfe359dac32
#float diffuse_color[3]; // dr, dg, db // 減衰色
#float alpha;
#float specularity;
#float specular_color[3]; // sr, sg, sb // 光沢色
#float mirror_color[3]; // mr, mg, mb // 環境色(ambient)
#BYTE toon_index; // toon??.bmp // 0.bmp:0xFF, 1(01).bmp:0x00 ・・・ 10.bmp:0x09
#BYTE edge_flag; // 輪郭、影
#DWORD face_vert_count; // 面頂点数 // インデックスに変換する場合は、材質0から順に加算
#char texture_file_name[20]; // テクスチャファイル名またはスフィアファイル名 // 20バイトぎりぎりまで使える(終端の0x00は無くても動く)
class Material
  constructor: (buffer, view, offset) ->
    tmp = []
    tmp[0] = view.getFloat32(offset, true); offset += size_Float32
    tmp[1] = view.getFloat32(offset, true); offset += size_Float32
    tmp[2] = view.getFloat32(offset, true); offset += size_Float32
    @diffuse = new Float32Array(tmp)
    @alpha = view.getFloat32(offset, true); offset += size_Float32
    @shininess = view.getFloat32(offset, true); offset += size_Float32
    tmp[0] = view.getFloat32(offset, true); offset += size_Float32
    tmp[1] = view.getFloat32(offset, true); offset += size_Float32
    tmp[2] = view.getFloat32(offset, true); offset += size_Float32
    @specular = new Float32Array(tmp)
    tmp[0] = view.getFloat32(offset, true); offset += size_Float32
    tmp[1] = view.getFloat32(offset, true); offset += size_Float32
    tmp[2] = view.getFloat32(offset, true); offset += size_Float32
    @ambient = new Float32Array(tmp)
    @toon_index = view.getInt8(offset); offset += size_Int8
    @edge_flag = view.getUint8(offset); offset += size_Uint8
    @face_vert_count = view.getUint32(offset, true); offset += size_Uint32
    @texture_file_name = sjisArrayToString(
      view.getUint8(offset + size_Uint8 * i) for i in [0...20])

Material.size = size_Float32 * 11 + size_Uint8 * 2 + size_Uint32 + size_Uint8 * 20 # 70

#http://blog.goo.ne.jp/torisu_tetosuki/e/638463f52d0ad6ca1c46fd315a9b17d0
#char bone_name[20]; // ボーン名
#WORD parent_bone_index; // 親ボーン番号(ない場合は0xFFFF)
#WORD tail_pos_bone_index; // tail位置のボーン番号(チェーン末端の場合は0xFFFF) // 親：子は1：多なので、主に位置決め用
#BYTE bone_type; // ボーンの種類
#WORD ik_parent_bone_index; // IKボーン番号(影響IKボーン。ない場合は0)
#float bone_head_pos[3]; // x, y, z // ボーンのヘッドの位置
class Bone
  constructor: (buffer, view, offset) ->
    @name = sjisArrayToString(new Uint8Array(buffer, offset, 20))
    offset += size_Uint8 * 20
    @parent_bone_index = view.getUint16(offset, true); offset += size_Uint16
    @tail_pos_bone_index = view.getUint16(offset, true); offset += size_Uint16
    @type = view.getUint8(offset); offset += size_Uint8
    @ik_parent_bone_index = view.getUint16(offset, true); offset += size_Uint16
    tmp = []
    tmp[0] = view.getFloat32(offset, true); offset += size_Float32
    tmp[1] = view.getFloat32(offset, true); offset += size_Float32
    tmp[2] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    @head_pos = new Float32Array(tmp)

Bone.size = size_Uint8 * 21 + size_Uint16 * 3 + size_Float32 * 3

#http://blog.goo.ne.jp/torisu_tetosuki/e/445cbbbe75c4b2622c22b473a27aaae9
#WORD ik_bone_index; // IKボーン番号
#WORD ik_target_bone_index; // IKターゲットボーン番号 // IKボーンが最初に接続するボーン
#BYTE ik_chain_length; // IKチェーンの長さ(子の数)
#WORD iterations; // 再帰演算回数 // IK値1
#float control_weight; // IKの影響度 // IK値2
#WORD ik_child_bone_index[ik_chain_length]; // IK影響下のボーン番号
class IK
  constructor: (buffer, view, offset) ->
    @bone_index = view.getUint16(offset, true); offset += size_Uint16
    @target_bone_index = view.getUint16(offset, true); offset += size_Uint16
    chain_length = view.getUint8(offset); offset += size_Uint8
    @iterations = view.getUint16(offset, true); offset += size_Uint16
    @control_weight = view.getFloat32(offset, true); offset += size_Float32
    @child_bones = (view.getUint16(offset + size_Uint16 * i, true) for i in [0...chain_length])
  getSize: ->
    size_Uint16 * 3 + size_Uint8 + size_Float32 + size_Uint16 * @child_bones.length

#http://blog.goo.ne.jp/torisu_tetosuki/e/8553151c445d261e122a3a31b0f91110
class Morph
  constructor: (buffer, view, offset) ->
    @name = sjisArrayToString(new Uint8Array(buffer, offset, 20))
    offset += size_Uint8 * 20
    vert_count = view.getUint32(offset, true); offset += size_Uint32
    @type = view.getUint8(offset); offset += size_Uint8
    @vert_data =
      for i in [0...vert_count]
        data = {}
        data.index = view.getUint32(offset, true); offset += size_Uint32
        data.x = view.getFloat32(offset, true); offset += size_Float32
        data.y = view.getFloat32(offset, true); offset += size_Float32
        data.z = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
        data

  getSize: ->
    size_Uint8 * 21 + size_Uint32 + (size_Uint32 + size_Float32 * 3) * @vert_data.length

#http://blog.goo.ne.jp/torisu_tetosuki/e/1e25fc196f2d7a7798f5cea87a942943
#char rigidbody_name[20]; // 諸データ：名称 // 頭
#WORD rigidbody_rel_bone_index; // 諸データ：関連ボーン番号 // 03 00 == 3 // 頭
#BYTE rigidbody_group_index; // 諸データ：グループ // 00
#WORD rigidbody_group_target; // 諸データ：グループ：対象 // 0xFFFFとの差 // 38 FE
#BYTE shape_type; // 形状：タイプ(0:球、1:箱、2:カプセル) // 00 // 球
#float shape_w; // 形状：半径(幅) // CD CC CC 3F // 1.6
#float shape_h; // 形状：高さ // CD CC CC 3D // 0.1
#float shape_d; // 形状：奥行 // CD CC CC 3D // 0.1
#float pos_pos[3]; // 位置：位置(x, y, z)
#float pos_rot[3]; // 位置：回転(rad(x), rad(y), rad(z))
#float rigidbody_weight; // 諸データ：質量 // 00 00 80 3F // 1.0
#float rigidbody_pos_dim; // 諸データ：移動減 // 00 00 00 00
#float rigidbody_rot_dim; // 諸データ：回転減 // 00 00 00 00
#float rigidbody_recoil; // 諸データ：反発力 // 00 00 00 00
#float rigidbody_friction; // 諸データ：摩擦力 // 00 00 00 00
#BYTE rigidbody_type; // 諸データ：タイプ(0:Bone追従、1:物理演算、2:物理演算(Bone位置合せ)) // 00 // Bone追従
class RigidBody
  constructor: (buffer, view, offset) ->
    @name = sjisArrayToString(new Uint8Array(buffer, offset, 20))
    offset += size_Uint8 * 20
    @rel_bone_index = view.getUint16(offset, true); offset += size_Uint16
    @group_index = view.getUint8(offset); offset += size_Uint8
    @group_target = view.getUint8(offset); offset += size_Uint8
    @shape_type = view.getUint8(offset, true); offset += size_Uint8
    @shape_w = view.getFloat32(offset, true); offset += size_Float32
    @shape_h = view.getFloat32(offset, true); offset += size_Float32
    @shape_d = view.getFloat32(offset, true); offset += size_Float32
    tmp = []
    tmp[0] = view.getFloat32(offset, true); offset += size_Float32
    tmp[1] = view.getFloat32(offset, true); offset += size_Float32
    tmp[2] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    @pos = new Float32Array(tmp)
    tmp[0] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[1] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[2] = view.getFloat32(offset, true); offset += size_Float32
    @rot = new Float32Array(tmp)
    @weight = view.getFloat32(offset, true); offset += size_Float32
    @pos_dim = view.getFloat32(offset, true); offset += size_Float32
    @rot_dim = view.getFloat32(offset, true); offset += size_Float32
    @recoil = view.getFloat32(offset, true); offset += size_Float32
    @friction = view.getFloat32(offset, true); offset += size_Float32
    @type = view.getUint8(offset); offset += size_Uint8

RigidBody.size = size_Uint8 * 23 + size_Uint16 * 2 + size_Float32 * 14

#http://blog.goo.ne.jp/torisu_tetosuki/e/b96dc839798f251ac235138b992a4481
#char joint_name[20]; // 諸データ：名称 // 右髪1
#DWORD joint_rigidbody_a; // 諸データ：剛体A
#DWORD joint_rigidbody_b; // 諸データ：剛体B
#float joint_pos[3]; // 諸データ：位置(x, y, z) // 諸データ：位置合せでも設定可
#float joint_rot[3]; // 諸データ：回転(rad(x), rad(y), rad(z))
#float constrain_pos_1[3]; // 制限：移動1(x, y, z)
#float constrain_pos_2[3]; // 制限：移動2(x, y, z)
#float constrain_rot_1[3]; // 制限：回転1(rad(x), rad(y), rad(z))
#float constrain_rot_2[3]; // 制限：回転2(rad(x), rad(y), rad(z))
#float spring_pos[3]; // ばね：移動(x, y, z)
#float spring_rot[3]; // ばね：回転(rad(x), rad(y), rad(z))

class Joint
  constructor: (buffer, view, offset) ->
    @name = sjisArrayToString(new Uint8Array(buffer, offset, 20))
    offset += size_Uint8 * 20
    @rigidbody_a = view.getUint32(offset, true); offset += size_Uint32
    @rigidbody_b = view.getUint32(offset, true); offset += size_Uint32
    tmp = []
    tmp[0] = view.getFloat32(offset, true); offset += size_Float32
    tmp[1] = view.getFloat32(offset, true); offset += size_Float32
    tmp[2] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    @pos = new Float32Array(tmp)
    tmp[0] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[1] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[2] = view.getFloat32(offset, true); offset += size_Float32
    @rot = new Float32Array(tmp)
    tmp[0] = view.getFloat32(offset, true); offset += size_Float32
    tmp[1] = view.getFloat32(offset, true); offset += size_Float32
    tmp[2] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    @constrain_pos_1 = new Float32Array(tmp)
    tmp[0] = view.getFloat32(offset, true); offset += size_Float32
    tmp[1] = view.getFloat32(offset, true); offset += size_Float32
    tmp[2] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    @constrain_pos_2 = new Float32Array(tmp)
    tmp[0] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[1] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[2] = view.getFloat32(offset, true); offset += size_Float32
    @constrain_rot_1 = new Float32Array(tmp)
    tmp[0] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[1] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[2] = view.getFloat32(offset, true); offset += size_Float32
    @constrain_rot_2 = new Float32Array(tmp)
    tmp[0] = view.getFloat32(offset, true); offset += size_Float32
    tmp[1] = view.getFloat32(offset, true); offset += size_Float32
    tmp[2] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    @spring_pos = new Float32Array(tmp)
    tmp[0] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[1] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[2] = view.getFloat32(offset, true); offset += size_Float32
    @spring_rot = new Float32Array(tmp)

Joint.size = size_Int8 * 20 + size_Uint32 * 2 + size_Float32 * 24

# see http://blog.goo.ne.jp/torisu_tetosuki/e/bc9f1c4d597341b394bd02b64597499d

# some shorthands
size_Uint8 = Uint8Array.BYTES_PER_ELEMENT
size_Uint32 = Uint32Array.BYTES_PER_ELEMENT
size_Float32 = Float32Array.BYTES_PER_ELEMENT

slice = Array.prototype.slice

class this.MMD.Motion # export to top level
  constructor: (path) ->
    @path = path

  load: (callback) ->
    xhr = new XMLHttpRequest
    xhr.open('GET', @path, true)
    xhr.responseType = 'arraybuffer'
    xhr.onload = =>
      console.time('parse')
      @parse(xhr.response)
      console.timeEnd('parse')
      callback()
    xhr.send()

  parse: (buffer) ->
    length = buffer.byteLength
    view = new DataView(buffer, 0)
    offset = 0
    offset = @checkHeader(buffer, view, offset)
    offset = @getModelName(buffer, view, offset)
    offset = @getBoneMotion(buffer, view, offset)
    offset = @getMorphMotion(buffer, view, offset)
    offset = @getCameraMotion(buffer, view, offset)
    offset = @getLightMotion(buffer, view, offset)
    offset = @getSelfShadowMotion(buffer, view, offset)

  checkHeader: (buffer, view, offset) ->
    if 'Vocaloid Motion Data 0002\0\0\0\0\0' !=
      String.fromCharCode.apply(null,
        slice.call(new Uint8Array(buffer, offset, 30)))
          throw 'File is not VMD'
    offset += 30 * size_Uint8

  getModelName: (buffer, view, offset) ->
    @model_name = sjisArrayToString(new Uint8Array(buffer, offset, 20))
    offset += size_Uint8 * 20

  getBoneMotion: (buffer, view, offset) ->
    length = view.getUint32(offset, true)
    offset += size_Uint32
    @bone =
      for i in [0...length]
        new BoneMotion(buffer, view, offset + i * BoneMotion.size)
    offset += length * BoneMotion.size

  getMorphMotion: (buffer, view, offset) ->
    length = view.getUint32(offset, true)
    offset += size_Uint32
    @morph =
      for i in [0...length]
        new MorphMotion(buffer, view, offset + i * MorphMotion.size)
    offset += length * MorphMotion.size

  getCameraMotion: (buffer, view, offset) ->
    length = view.getUint32(offset, true)
    offset += size_Uint32
    @camera =
      for i in [0...length]
        new CameraMotion(buffer, view, offset + i * CameraMotion.size)
    offset += length * CameraMotion.size

  getLightMotion: (buffer, view, offset) ->
    length = view.getUint32(offset, true)
    offset += size_Uint32
    @light =
      for i in [0...length]
        new LightMotion(buffer, view, offset + i * LightMotion.size)
    offset += length * LightMotion.size

  getSelfShadowMotion: (buffer, view, offset) ->
    length = view.getUint32(offset, true)
    offset += size_Uint32
    @selfshadow =
      for i in [0...length]
        new SelfShadowMotion(buffer, view, offset + i * SelfShadowMotion.size)
    offset += length * SelfShadowMotion.size


#char BoneName[15];
#DWORD FlameNo;
#float Location[3];
#float Rotatation[4]; // Quaternion
#BYTE Interpolation[64]; // [4][4][4]
class BoneMotion
  constructor: (buffer, view, offset) ->
    @name = sjisArrayToString(new Uint8Array(buffer, offset, 15))
    offset += size_Uint8 * 15
    @frame = view.getUint32(offset, true); offset += size_Uint32
    tmp = []
    tmp[0] = view.getFloat32(offset, true); offset += size_Float32
    tmp[1] = view.getFloat32(offset, true); offset += size_Float32
    tmp[2] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    @location = new Float32Array(tmp)
    tmp[0] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[1] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[2] = view.getFloat32(offset, true); offset += size_Float32
    tmp[3] = view.getFloat32(offset, true); offset += size_Float32
    @rotation = new Float32Array(tmp)
    for i in [0...64]
      tmp[i] = view.getUint8(offset, true); offset += size_Uint8
    @interpolation = new Uint8Array(tmp)

BoneMotion.size = size_Uint8 * (15 + 64) + size_Uint32 + size_Float32 * 7

#char SkinName[15];
#DWORD FlameNo;
#float Weight;
class MorphMotion
  constructor: (buffer, view, offset) ->
    @name = sjisArrayToString(new Uint8Array(buffer, offset, 15))
    offset += size_Uint8 * 15
    @frame = view.getUint32(offset, true); offset += size_Uint32
    @weight = view.getFloat32(offset, true); offset += size_Float32

MorphMotion.size = size_Uint8 * 15 + size_Uint32 + size_Float32

#DWORD FlameNo;
#float Length; // -(距離)
#float Location[3];
#float Rotation[3]; // オイラー角 // X軸は符号が反転しているので注意
#BYTE Interpolation[24]; // おそらく[6][4](未検証)
#DWORD ViewingAngle;
#BYTE Perspective; // 0:on 1:off
class CameraMotion
  constructor: (buffer, view, offset) ->
    @frame = view.getUint32(offset, true); offset += size_Uint32
    @distance = - view.getFloat32(offset, true); offset += size_Float32
    tmp = []
    tmp[0] = view.getFloat32(offset, true); offset += size_Float32
    tmp[1] = view.getFloat32(offset, true); offset += size_Float32
    tmp[2] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    @location = new Float32Array(tmp)
    tmp[0] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[1] = - view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    tmp[2] = view.getFloat32(offset, true); offset += size_Float32
    @rotation = new Float32Array(tmp)
    for i in [0...24]
      tmp[i] = view.getUint8(offset, true); offset += size_Uint8
    @interpolation = new Uint8Array(tmp)
    @view_angle = view.getUint32(offset, true); offset += size_Uint32
    @noPerspective = view.getUint8(offset, true); offset += size_Uint8

CameraMotion.size = size_Float32 * 7 + size_Uint8 * 25 + size_Float32 * 2

#DWORD FlameNo;
#float RGB[3]; // RGB各値/256
#float Location[3];
class LightMotion
  constructor: (buffer, view, offset) ->
    @frame = view.getUint32(offset, true); offset += size_Uint32
    tmp = []
    tmp[0] = view.getFloat32(offset, true); offset += size_Float32
    tmp[1] = view.getFloat32(offset, true); offset += size_Float32
    tmp[2] = view.getFloat32(offset, true); offset += size_Float32
    @color = new Float32Array(tmp)
    tmp = []
    tmp[0] = - view.getFloat32(offset, true); offset += size_Float32
    tmp[1] = - view.getFloat32(offset, true); offset += size_Float32
    tmp[2] = view.getFloat32(offset, true); offset += size_Float32 # left->right handed system
    @location = new Float32Array(tmp)

LightMotion.size = size_Float32 * 6 + size_Uint32

#DWORD FlameNo;
#BYTE Mode; // 00-02
#float Distance; // 0.1 - (dist * 0.00001)
class SelfShadowMotion
  constructor: (buffer, view, offset) ->
    @frame = view.getUint32(offset, true); offset += size_Uint32
    @mode = view.getUint8(offset, true); offset += size_Uint8
    @distance = view.getFloat32(offset, true); offset += size_Float32

SelfShadowMotion.size = size_Float32 + size_Uint8 + size_Float32


class MMD.MotionManager
  constructor: ->
    @modelMotions = []
    @cameraMotion = []
    @cameraFrames = []
    @lightMotion = []
    @lightFrames = []
    @lastFrame = 0
    return

  addModelMotion: (model, motion, merge_flag, frame_offset) ->
    for mm, i in @modelMotions
      break if model == mm.model

    if i == @modelMotions.length
      mm = new ModelMotion(model)
      @modelMotions.push(mm)

    mm.addBoneMotion(motion.bone, merge_flag, frame_offset)
    mm.addMorphMotion(motion.morph, merge_flag, frame_offset)

    @lastFrame = mm.lastFrame
    return

  getModelFrame: (model, frame) ->
    for mm, i in @modelMotions
      break if model == mm.model

    return {} if i == @modelMotions.length

    return {
      bones: mm.getBoneFrame(frame)
      morphs: mm.getMorphFrame(frame)
    }

  addCameraLightMotion: (motion, merge_flag, frame_offset) ->
    @addCameraMotion(motion.camera, merge_flag, frame_offset)
    @addLightMotoin(motion.light, merge_flag, frame_offset)
    return

  addCameraMotion: (camera, merge_flag, frame_offset) ->
    return if camera.length == 0
    if not merge_flag
      @cameraMotion = []
      @cameraFrames = []

    frame_offset = frame_offset || 0

    for c in camera
      frame = c.frame + frame_offset
      @cameraMotion[frame] = c
      @cameraFrames.push(frame)
      @lastFrame = frame if @lastFrame < frame
    @cameraFrames = @cameraFrames.sort((a, b) -> a - b)
    return

  addLightMotoin: (light, merge_flag, frame_offset) ->
    return if light.length == 0
    if not merge_flag
      @lightMotion = []
      @lightFrames = []

    frame_offset = frame_offset || 0

    for l in light
      frame = l.frame + frame_offset
      @lightMotion[frame] = l
      @lightFrames.push(frame)
      @lastFrame = frame if @lastFrame < frame
    @lightFrames = @lightFrames.sort((a, b) -> a - b)
    return

  getCameraFrame: (frame) ->
    return null if not @cameraMotion.length
    timeline = @cameraMotion
    frames = @cameraFrames
    lastFrame = frames[frames.length - 1]
    if lastFrame <= frame
      camera = timeline[lastFrame]
    else
      idx = previousRegisteredFrame(frames, frame)
      p = frames[idx]
      n = frames[idx + 1]
      frac = fraction(frame, p, n)
      prev = timeline[p] # previous registered frame
      next = timeline[n] # next registered frame

      cache = []
      bez = (i)->
        X1 = next.interpolation[i * 4    ]
        X2 = next.interpolation[i * 4 + 1]
        Y1 = next.interpolation[i * 4 + 2]
        Y2 = next.interpolation[i * 4 + 3]
        id = X1 | (X2 << 8) | (Y1 << 16) | (Y2 << 24)
        return cache[id] if cache[id]?
        return cache[id] = frac if X1 == Y1 and X2 == Y2
        return cache[id] = bezierp(X1 / 127, X2 / 127, Y1 / 127, Y2 / 127, frac)

      camera = {
        location: vec3.createLerp3(prev.location, next.location, [bez(0), bez(1), bez(2)])
        rotation: vec3.createLerp(prev.rotation, next.rotation, bez(3))
        distance: lerp1(prev.distance, next.distance, bez(4))
        view_angle: lerp1(prev.view_angle, next.view_angle, bez(5))
      }

    return camera

  getLightFrame: (frame) ->
    return null if not @lightMotion.length
    timeline = @lightMotion
    frames = @lightFrames
    lastFrame = frames[frames.length - 1]
    if lastFrame <= frame
      light = timeline[lastFrame]
    else
      idx = previousRegisteredFrame(frames, frame)
      p = frames[idx]
      n = frames[idx + 1]
      frac = fraction(frame, p, n)
      prev = timeline[p] # previous registered frame
      next = timeline[n] # next registered frame

      light = {
        color: vec3.createLerp(prev.color, next.color, frac)
        location: vec3.lerp(prev.location, next.location, frac)
      }

    return light

class ModelMotion
  constructor: (@model) ->
    @boneMotions = {}
    @boneFrames = {}
    @morphMotions = {}
    @morphFrames = {}
    @lastFrame = 0

  addBoneMotion: (bone, merge_flag, frame_offset) ->
    if not merge_flag
      @boneMotions = {}
      @boneFrames = {}

    frame_offset = frame_offset || 0

    for b in bone
      if not @boneMotions[b.name] # set 0th frame
        @boneMotions[b.name] = [{location: vec3.create(), rotation: quat4.create([0, 0, 0, 1])}]

      frame = b.frame + frame_offset
      @boneMotions[b.name][frame] = b
      @lastFrame = frame if @lastFrame < frame

    for name of @boneMotions
      @boneFrames[name] = (@boneFrames[name] || []).
        concat(Object.keys(@boneMotions[name]).map(Number)).sort((a,b) -> a - b)

    return

  addMorphMotion: (morph, merge_flag, frame_offset) ->
    if not merge_flag
      @morphMotions = {}
      @morphFrames = {}

    frame_offset = frame_offset || 0

    for m in morph
      continue if m.name == 'base'
      @morphMotions[m.name] = [0] if !@morphMotions[m.name] # set 0th frame as 0
      frame = m.frame + frame_offset
      @morphMotions[m.name][frame] = m.weight
      @lastFrame = frame if @lastFrame < frame

    for name of @morphMotions
      @morphFrames[name] = (@morphFrames[name] || []).
        concat(Object.keys(@morphMotions[name]).map(Number)).sort((a,b) -> a - b)

    return

  getBoneFrame: (frame) ->
    bones = {}

    for name of @boneMotions
      timeline = @boneMotions[name]
      frames = @boneFrames[name]
      lastFrame = frames[frames.length - 1]
      if lastFrame <= frame
        bones[name] = timeline[lastFrame]
      else
        idx = previousRegisteredFrame(frames, frame)
        p = frames[idx]
        n = frames[idx + 1]
        frac = fraction(frame, p, n)
        prev = timeline[p]
        next = timeline[n]

        cache = []
        bez = (i)->
          X1 = next.interpolation[i * 4    ]
          X2 = next.interpolation[i * 4 + 1]
          Y1 = next.interpolation[i * 4 + 2]
          Y2 = next.interpolation[i * 4 + 3]
          id = X1 | (X2 << 8) | (Y1 << 16) | (Y2 << 24)
          return cache[id] if cache[id]?
          return cache[id] = frac if X1 == Y1 and X2 == Y2
          return cache[id] = bezierp(X1 / 127, X2 / 127, Y1 / 127, Y2 / 127, frac)

        if quat4.dot(prev.rotation, next.rotation) >= 0
          rotation = quat4.createSlerp(prev.rotation, next.rotation, bez(3))
        else
          r = prev.rotation
          rotation = quat4.createSlerp([-r[0], -r[1], -r[2], -r[3]], next.rotation, bez(3))

        bones[name] = {
          location: vec3.createLerp3(prev.location, next.location, [bez(0), bez(1), bez(2)])
          rotation: rotation
        }

    return bones

  getMorphFrame: (frame) ->
    morphs = {}

    for name of @morphMotions
      timeline = @morphMotions[name]
      frames = @morphFrames[name]
      lastFrame = frames[frames.length - 1]
      if lastFrame <= frame
        morphs[name] = timeline[lastFrame]
      else
        idx = previousRegisteredFrame(frames, frame)
        p = frames[idx]
        n = frames[idx + 1]
        frac = fraction(frame, p, n)
        prev = timeline[p] # previous registered frame
        next = timeline[n] # next registered frame

        morphs[name] = lerp1(prev, next, frac)

    return morphs


# utils
previousRegisteredFrame = (frames, frame) ->
  ###
    'frames' is key frames registered, 'frame' is the key frame I'm enquiring about
    ex. frames: [0,10,20,30,40,50], frame: 15
    now I want to find the numbers 10 and 20, namely the ones before 15 and after 15
    I'm doing a bisection search here.
  ###
  idx = 0
  delta = frames.length
  while true
    delta = (delta >> 1) || 1
    if frames[idx] <= frame
      break if delta == 1 and frames[idx + 1] > frame
      idx += delta
    else
      idx -= delta
      break if delta == 1 and frames[idx] <= frame
  return idx

fraction = (x, x0, x1) ->
  return (x - x0) / (x1 - x0)

lerp1 = (x0, x1, a) ->
  return x0 + a * (x1 - x0)

bezierp = (x1, x2, y1, y2, x) ->
  ###
    interpolate using Bezier curve (http://musashi.or.tv/fontguide_doc3.htm)
    Bezier curve is parametrized by t (0 <= t <= 1)
      x = s^3 x_0 + 3 s^2 t x_1 + 3 s t^2 x_2 + t^3 x_3
      y = s^3 y_0 + 3 s^2 t y_1 + 3 s t^2 y_2 + t^3 y_3
    where s is defined as s = 1 - t.
    Especially, for MMD, (x_0, y_0) = (0, 0) and (x_3, y_3) = (1, 1), so
      x = 3 s^2 t x_1 + 3 s t^2 x_2 + t^3
      y = 3 s^2 t y_1 + 3 s t^2 y_2 + t^3
    Now, given x, find t by bisection method (http://en.wikipedia.org/wiki/Bisection_method)
    i.e. find t such that f(t) = 3 s^2 t x_1 + 3 s t^2 x_2 + t^3 - x = 0
    One thing to note here is that f(t) is monotonically increasing in the range [0,1]
    Therefore, when I calculate f(t) for the t I guessed,
    Finally find y for the t.
  ###
  #Adopted from MMDAgent
  t = x
  while true
    v = ipfunc(t, x1, x2) - x
    break if v * v < 0.0000001 # Math.abs(v) < 0.0001
    tt = ipfuncd(t, x1, x2)
    break if tt == 0
    t -= v / tt
  return ipfunc(t, y1, y2)

ipfunc = (t, p1, p2) ->
  ((1 + 3 * p1 - 3 * p2) * t * t * t + (3 * p2 - 6 * p1) * t * t + 3 * p1 * t)

ipfuncd = (t, p1, p2) ->
  ((3 + 9 * p1 - 9 * p2) * t * t + (6 * p2 - 12 * p1) * t + 3 * p1)


class MMD.ShadowMap
  constructor: (mmd) ->
    @mmd = mmd
    @framebuffer = @texture = null
    @width = @height = 2048
    @viewBroadness = 0.6
    @debug = false

    @initFramebuffer()

  initFramebuffer: ->
    gl = @mmd.gl
    @framebuffer = gl.createFramebuffer()
    gl.bindFramebuffer(gl.FRAMEBUFFER, @framebuffer)

    @texture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, @texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST)
    gl.generateMipmap(gl.TEXTURE_2D)

    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, @width, @height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)

    renderbuffer = gl.createRenderbuffer()
    gl.bindRenderbuffer(gl.RENDERBUFFER, renderbuffer)
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, @width, @height)

    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, @texture, 0)
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, renderbuffer)

    gl.bindTexture(gl.TEXTURE_2D, null)
    gl.bindRenderbuffer(gl.RENDERBUFFER, null)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)

  computeMatrices: ->
    # from mmd's vectors and matrices, calculate the "light" space's transform matrices

    center = vec3.create(@mmd.center) # center of view in world space

    lightDirection = vec3.createNormalize(@mmd.lightDirection) # becomes the camera direction in light space
    vec3.add(lightDirection, center)

    cameraPosition = vec3.create(@mmd.cameraPosition)
    lengthScale = vec3.lengthBetween(cameraPosition, center)
    size = lengthScale * this.viewBroadness # size of shadowmap

    viewMatrix = mat4.lookAt(lightDirection, center, [0, 1, 0])

    @mvMatrix = mat4.createMultiply(viewMatrix, @mmd.modelMatrix)

    mat4.multiplyVec3(viewMatrix, center) # transform center in view space
    cx = center[0]; cy = center[1]
    @pMatrix = mat4.ortho(cx - size, cx + size, cy - size, cy + size, -size, size) # orthographic projection; near can be negative
    return

  beforeRender: ->
    gl = @mmd.gl
    program = @mmd.program

    gl.bindFramebuffer(gl.FRAMEBUFFER, @framebuffer)

    gl.viewport(0, 0, @width, @height)
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

    gl.uniform1i(program.uGenerateShadowMap, true)
    gl.uniformMatrix4fv(program.uMVMatrix, false, @mvMatrix)
    gl.uniformMatrix4fv(program.uPMatrix, false, @pMatrix)

    return

  afterRender: ->
    gl = @mmd.gl
    program = @mmd.program

    gl.uniform1i(program.uGenerateShadowMap, false)

    gl.bindTexture(gl.TEXTURE_2D, @texture)
    gl.generateMipmap(gl.TEXTURE_2D)
    gl.bindTexture(gl.TEXTURE_2D, null)
    @debugTexture() if (@debug)
    return

  getLightMatrix: ->
    # display matrix transforms projection space to screen space. in fragment shader screen coordinates are available as gl_FragCoord
    # http://www.c3.club.kyutech.ac.jp/gamewiki/index.php?3D%BA%C2%C9%B8%CA%D1%B4%B9
    lightMatrix = mat4.createMultiply(@pMatrix, @mvMatrix)
    mat4.applyScale(lightMatrix, [0.5, 0.5, 0.5])
    mat4.applyTranslate(lightMatrix, [0.5, 0.5, 0.5])
    return lightMatrix

  debugTexture: ->
    gl = @mmd.gl
    pixelarray = new Uint8Array(@width * @height * 4)
    gl.readPixels(0, 0, @width, @height, gl.RGBA, gl.UNSIGNED_BYTE, pixelarray)

    canvas = document.getElementById('shadowmap')
    if not canvas
      canvas = document.createElement('canvas')
      canvas.id = 'shadowmap'
      canvas.width = @width
      canvas.height = @height
      canvas.style.border = 'solid black 1px'
      canvas.style.width = @mmd.width + 'px'
      canvas.style.height = @mmd.height + 'px'
      document.body.appendChild(canvas)

    ctx = canvas.getContext('2d')
    imageData = ctx.getImageData(0, 0, @width, @height)
    data = imageData.data
    data[i] = pixelarray[i] for i in [0...data.length]
    ctx.putImageData(imageData, 0, 0)

  getTexture: ->
    @texture

class MMD.TextureManager
  constructor: (mmd) ->
    @mmd = mmd
    @store = {}
    @pendingCount = 0

  get: (type, url) ->
    texture = @store[url]
    return texture if texture

    gl = @mmd.gl
    texture = @store[url] = gl.createTexture()

    loadImage(url, (img) =>
      img = checkSize(img)

      gl.bindTexture(gl.TEXTURE_2D, texture)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR)

      if type == 'toon'
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      else
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT)

      gl.generateMipmap(gl.TEXTURE_2D)
      gl.bindTexture(gl.TEXTURE_2D, null)

      @onload(img) if @onload
      --@pendingCount
    )
    @pendingCount++

    return texture

# utils
checkSize = (img) ->
  w = img.naturalWidth
  h = img.naturalHeight
  size = 1 << (Math.log(Math.min(w, h)) / Math.LN2 | 0) # largest 2^n integer that don't exceed w or h
  if w != h || w != size
    canv = document.createElement('canvas')
    canv.height = canv.width = size
    canv.getContext('2d').drawImage(img, 0, 0, w, h, 0, 0, size, size)
    img = canv

  return img

loadImage = (url, callback) ->
  img = new Image
  img.onload = -> callback(img)
  img.onerror = -> alert('failed to load image: ' + url)
  img.src = url

  return img

MMD.VertexShaderSource = '''

  uniform mat4 uMVMatrix; // model-view matrix (model -> view space)
  uniform mat4 uPMatrix; // projection matrix (view -> projection space)
  uniform mat4 uNMatrix; // normal matrix (inverse of transpose of model-view matrix)

  uniform mat4 uLightMatrix; // mvpdMatrix of light space (model -> display space)

  attribute vec3 aVertexNormal;
  attribute vec2 aTextureCoord;
  attribute float aVertexEdge; // 0 or 1. 1 if the vertex has an edge. (becuase we can't pass bool to attributes)

  attribute float aBoneWeight;
  attribute vec3 aVectorFromBone1;
  attribute vec3 aVectorFromBone2;
  attribute vec4 aBone1Rotation;
  attribute vec4 aBone2Rotation;
  attribute vec3 aBone1Position;
  attribute vec3 aBone2Position;

  attribute vec3 aMultiPurposeVector;

  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec2 vTextureCoord;
  varying vec4 vLightCoord; // coordinate in light space; to be mapped onto shadow map

  uniform float uEdgeThickness;
  uniform bool uEdge;

  uniform bool uGenerateShadowMap;

  uniform bool uSelfShadow;

  uniform bool uAxis;
  uniform bool uCenterPoint;

  vec3 qtransform(vec4 q, vec3 v) {
    return v + 2.0 * cross(cross(v, q.xyz) - q.w*v, q.xyz);
  }

  void main() {
    vec3 position;
    vec3 normal;

    if (uAxis || uCenterPoint) {

      position = aMultiPurposeVector;

    } else {

      float weight = aBoneWeight;
      vec3 morph = aMultiPurposeVector;

      position = qtransform(aBone1Rotation, aVectorFromBone1 + morph) + aBone1Position;
      normal = qtransform(aBone1Rotation, aVertexNormal);

      if (weight < 0.99) {
        vec3 p2 = qtransform(aBone2Rotation, aVectorFromBone2 + morph) + aBone2Position;
        vec3 n2 = qtransform(aBone2Rotation, normal);

        position = mix(p2, position, weight);
        normal = normalize(mix(n2, normal, weight));
      }
    }

    // return vertex point in projection space
    gl_Position = uPMatrix * uMVMatrix * vec4(position, 1.0);

    if (uCenterPoint) {
      gl_Position.z = 0.0; // always on top
      gl_PointSize = 16.0;
    }

    if (uGenerateShadowMap || uAxis || uCenterPoint) return;

    // for fragment shader
    vTextureCoord = aTextureCoord;
    vPosition = (uMVMatrix * vec4(position, 1.0)).xyz;
    vNormal = (uNMatrix * vec4(normal, 1.0)).xyz;

    if (uSelfShadow) {
      vLightCoord = uLightMatrix * vec4(position, 1.0);
    }

    if (uEdge) {
      vec4 pos = gl_Position;
      vec4 pos2 = uPMatrix * uMVMatrix * vec4(position + normal, 1.0);
      vec4 norm = normalize(pos2 - pos);
      gl_Position = pos + norm * uEdgeThickness * aVertexEdge * pos.w; // scale by pos.w to prevent becoming thicker when zoomed
      return;
    }
  }

'''
