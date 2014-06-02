var Bone, BoneMotion, CameraMotion, IK, Joint, LightMotion, Material, ModelMotion, Morph, MorphMotion, RigidBody, SelfShadowMotion, Vertex, bezierp, checkSize, fraction, ipfunc, ipfuncd, lerp1, loadImage, previousRegisteredFrame, size_Float32, size_Int8, size_Uint16, size_Uint32, size_Uint8, slice,
  __indexOf = [].indexOf || function(item) { for (var i = 0, l = this.length; i < l; i++) { if (i in this && this[i] === item) return i; } return -1; };

MMD.FragmentShaderSource = '\n#ifdef GL_ES\nprecision highp float;\n#endif\n\nvarying vec2 vTextureCoord;\nvarying vec3 vPosition;\nvarying vec3 vNormal;\nvarying vec4 vLightCoord;\n\nuniform vec3 uLightDirection; // light source direction in world space\nuniform vec3 uLightColor;\n\nuniform vec3 uAmbientColor;\nuniform vec3 uSpecularColor;\nuniform vec3 uDiffuseColor;\nuniform float uAlpha;\nuniform float uShininess;\n\nuniform bool uUseTexture;\nuniform bool uUseSphereMap;\nuniform bool uIsSphereMapAdditive;\n\nuniform sampler2D uToon;\nuniform sampler2D uTexture;\nuniform sampler2D uSphereMap;\n\nuniform bool uEdge;\nuniform float uEdgeThickness;\nuniform vec3 uEdgeColor;\n\nuniform bool uGenerateShadowMap;\nuniform bool uSelfShadow;\nuniform sampler2D uShadowMap;\n\nuniform bool uAxis;\nuniform vec3 uAxisColor;\nuniform bool uCenterPoint;\n\n// from http://spidergl.org/example.php?id=6\nvec4 pack_depth(const in float depth) {\n  const vec4 bit_shift = vec4(256.0*256.0*256.0, 256.0*256.0, 256.0, 1.0);\n  const vec4 bit_mask  = vec4(0.0, 1.0/256.0, 1.0/256.0, 1.0/256.0);\n  vec4 res = fract(depth * bit_shift);\n  res -= res.xxyz * bit_mask;\n  return res;\n}\nfloat unpack_depth(const in vec4 rgba_depth)\n{\n  const vec4 bit_shift = vec4(1.0/(256.0*256.0*256.0), 1.0/(256.0*256.0), 1.0/256.0, 1.0);\n  float depth = dot(rgba_depth, bit_shift);\n  return depth;\n}\n\nvoid main() {\n  if (uGenerateShadowMap) {\n    //gl_FragData[0] = pack_depth(gl_FragCoord.z);\n    gl_FragColor = pack_depth(gl_FragCoord.z);\n    return;\n  }\n  if (uAxis) {\n    gl_FragColor = vec4(uAxisColor, 1.0);\n    return;\n  }\n  if (uCenterPoint) {\n    vec2 uv = gl_PointCoord * 2.0 - 1.0; // transform [0, 1] -> [-1, 1] coord systems\n    float w = dot(uv, uv);\n    if (w < 0.3 || (w > 0.5 && w < 1.0)) {\n      gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);\n    } else {\n      discard;\n    }\n    return;\n  }\n\n  // vectors are in view space\n  vec3 norm = normalize(vNormal); // each point\'s normal vector in view space\n  vec3 cameraDirection = normalize(-vPosition); // camera located at origin in view space\n\n  vec3 color;\n  float alpha = uAlpha;\n\n  if (uEdge) {\n\n    color = uEdgeColor;\n\n  } else {\n\n    color = vec3(1.0, 1.0, 1.0);\n    if (uUseTexture) {\n      vec4 texColor = texture2D(uTexture, vTextureCoord);\n      color *= texColor.rgb;\n      alpha *= texColor.a;\n    }\n    if (uUseSphereMap) {\n      vec2 sphereCoord = 0.5 * (1.0 + vec2(1.0, -1.0) * norm.xy);\n      if (uIsSphereMapAdditive) {\n        color += texture2D(uSphereMap, sphereCoord).rgb;\n      } else {\n        color *= texture2D(uSphereMap, sphereCoord).rgb;\n      }\n    }\n\n    // specular component\n    vec3 halfAngle = normalize(uLightDirection + cameraDirection);\n    float specularWeight = pow( max(0.001, dot(halfAngle, norm)) , uShininess );\n    //float specularWeight = pow( max(0.0, dot(reflect(-uLightDirection, norm), cameraDirection)) , uShininess ); // another definition\n    vec3 specular = specularWeight * uSpecularColor;\n\n    vec2 toonCoord = vec2(0.0, 0.5 * (1.0 - dot( uLightDirection, norm )));\n\n    if (uSelfShadow) {\n      vec3 lightCoord = vLightCoord.xyz / vLightCoord.w; // projection to texture coordinate (in light space)\n      vec4 rgbaDepth = texture2D(uShadowMap, lightCoord.xy);\n      float depth = unpack_depth(rgbaDepth);\n      if (depth < lightCoord.z - 0.01) {\n        toonCoord = vec2(0.0, 0.55);\n      }\n    }\n\n    color *= uAmbientColor + uLightColor * (uDiffuseColor + specular);\n\n    color = clamp(color, 0.0, 1.0);\n    color *= texture2D(uToon, toonCoord).rgb;\n\n  }\n  gl_FragColor = vec4(color, alpha);\n\n}\n';

size_Int8 = Int8Array.BYTES_PER_ELEMENT;

size_Uint8 = Uint8Array.BYTES_PER_ELEMENT;

size_Uint16 = Uint16Array.BYTES_PER_ELEMENT;

size_Uint32 = Uint32Array.BYTES_PER_ELEMENT;

size_Float32 = Float32Array.BYTES_PER_ELEMENT;

slice = Array.prototype.slice;

this.MMD.Model = (function() {
  function Model(directory, filename) {
    this.directory = directory;
    this.filename = filename;
    this.vertices = null;
    this.triangles = null;
    this.materials = null;
    this.bones = null;
    this.morphs = null;
    this.morph_order = null;
    this.bone_group_names = null;
    this.bone_table = null;
    this.english_flag = null;
    this.english_name = null;
    this.english_comment = null;
    this.english_bone_names = null;
    this.english_morph_names = null;
    this.english_bone_group_names = null;
    this.toon_file_names = null;
    this.rigid_bodies = null;
    this.joints = null;
  }

  Model.prototype.load = function(callback) {
    var xhr;
    xhr = new XMLHttpRequest;
    xhr.open('GET', this.directory + '/' + this.filename, true);
    xhr.responseType = 'arraybuffer';
    xhr.onload = (function(_this) {
      return function() {
        console.time('parse');
        _this.parse(xhr.response);
        console.timeEnd('parse');
        return callback();
      };
    })(this);
    return xhr.send();
  };

  Model.prototype.parse = function(buffer) {
    var length, offset, view;
    length = buffer.byteLength;
    view = new DataView(buffer, 0);
    offset = 0;
    offset = this.checkHeader(buffer, view, offset);
    offset = this.getName(buffer, view, offset);
    offset = this.getVertices(buffer, view, offset);
    offset = this.getTriangles(buffer, view, offset);
    offset = this.getMaterials(buffer, view, offset);
    offset = this.getBones(buffer, view, offset);
    offset = this.getIKs(buffer, view, offset);
    offset = this.getMorphs(buffer, view, offset);
    offset = this.getMorphOrder(buffer, view, offset);
    offset = this.getBoneGroupNames(buffer, view, offset);
    offset = this.getBoneTable(buffer, view, offset);
    if (offset >= length) {
      return;
    }
    offset = this.getEnglishFlag(buffer, view, offset);
    if (this.english_flag) {
      offset = this.getEnglishName(buffer, view, offset);
      offset = this.getEnglishBoneNames(buffer, view, offset);
      offset = this.getEnglishMorphNames(buffer, view, offset);
      offset = this.getEnglishBoneGroupNames(buffer, view, offset);
    }
    if (offset >= length) {
      return;
    }
    offset = this.getToonFileNames(buffer, view, offset);
    if (offset >= length) {
      return;
    }
    offset = this.getRigidBodies(buffer, view, offset);
    return offset = this.getJoints(buffer, view, offset);
  };

  Model.prototype.checkHeader = function(buffer, view, offset) {
    if (view.getUint8(0) !== 'P'.charCodeAt(0) || view.getUint8(1) !== 'm'.charCodeAt(0) || view.getUint8(2) !== 'd'.charCodeAt(0) || view.getUint8(3) !== 0x00 || view.getUint8(4) !== 0x00 || view.getUint8(5) !== 0x80 || view.getUint8(6) !== 0x3F) {
      throw 'File is not PMD';
    }
    return offset += 7 * size_Uint8;
  };

  Model.prototype.getName = function(buffer, view, offset) {
    var block;
    block = new Uint8Array(buffer, offset, 20 + 256);
    this.name = sjisArrayToString(slice.call(block, 0, 20));
    this.comment = sjisArrayToString(slice.call(block, 20, 20 + 256));
    return offset += (20 + 256) * size_Uint8;
  };

  Model.prototype.getVertices = function(buffer, view, offset) {
    var i, length;
    length = view.getUint32(offset, true);
    offset += size_Uint32;
    this.vertices = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(new Vertex(buffer, view, offset + i * Vertex.size));
      }
      return _results;
    })();
    return offset += length * Vertex.size;
  };

  Model.prototype.getTriangles = function(buffer, view, offset) {
    var i, length, _i;
    length = view.getUint32(offset, true);
    offset += size_Uint32;
    this.triangles = new Uint16Array(length);
    for (i = _i = 0; _i < length; i = _i += 3) {
      this.triangles[i + 1] = view.getUint16(offset + i * size_Uint16, true);
      this.triangles[i] = view.getUint16(offset + (i + 1) * size_Uint16, true);
      this.triangles[i + 2] = view.getUint16(offset + (i + 2) * size_Uint16, true);
    }
    return offset += length * size_Uint16;
  };

  Model.prototype.getMaterials = function(buffer, view, offset) {
    var i, length;
    length = view.getUint32(offset, true);
    offset += size_Uint32;
    this.materials = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(new Material(buffer, view, offset + i * Material.size));
      }
      return _results;
    })();
    return offset += length * Material.size;
  };

  Model.prototype.getBones = function(buffer, view, offset) {
    var i, length;
    length = view.getUint16(offset, true);
    offset += size_Uint16;
    this.bones = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(new Bone(buffer, view, offset + i * Bone.size));
      }
      return _results;
    })();
    return offset += length * Bone.size;
  };

  Model.prototype.getIKs = function(buffer, view, offset) {
    var i, ik, length;
    length = view.getUint16(offset, true);
    offset += size_Uint16;
    this.iks = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        ik = new IK(buffer, view, offset);
        offset += ik.getSize();
        _results.push(ik);
      }
      return _results;
    })();
    return offset;
  };

  Model.prototype.getMorphs = function(buffer, view, offset) {
    var i, length, morph;
    length = view.getUint16(offset, true);
    offset += size_Uint16;
    this.morphs = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        morph = new Morph(buffer, view, offset);
        offset += morph.getSize();
        _results.push(morph);
      }
      return _results;
    })();
    return offset;
  };

  Model.prototype.getMorphOrder = function(buffer, view, offset) {
    var i, length;
    length = view.getUint8(offset);
    offset += size_Uint8;
    this.morph_order = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(view.getUint16(offset + i * size_Uint16, true));
      }
      return _results;
    })();
    return offset += length * size_Uint16;
  };

  Model.prototype.getBoneGroupNames = function(buffer, view, offset) {
    var block, i, length;
    length = view.getUint8(offset);
    offset += size_Uint8;
    block = new Uint8Array(buffer, offset, 50 * length);
    this.bone_group_names = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(sjisArrayToString(slice.call(block, i * 50, (i + 1) * 50)));
      }
      return _results;
    })();
    return offset += length * 50 * size_Uint8;
  };

  Model.prototype.getBoneTable = function(buffer, view, offset) {
    var bone, i, length;
    length = view.getUint32(offset, true);
    offset += size_Uint32;
    this.bone_table = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        bone = {};
        bone.index = view.getUint16(offset, true);
        offset += size_Uint16;
        bone.group_index = view.getUint8(offset);
        offset += size_Uint8;
        _results.push(bone);
      }
      return _results;
    })();
    return offset;
  };

  Model.prototype.getEnglishFlag = function(buffer, view, offset) {
    this.english_flag = view.getUint8(offset);
    return offset += size_Uint8;
  };

  Model.prototype.getEnglishName = function(buffer, view, offset) {
    var block;
    block = new Uint8Array(buffer, offset, 20 + 256);
    this.english_name = sjisArrayToString(slice.call(block, 0, 20));
    this.english_comment = sjisArrayToString(slice.call(block, 20, 20 + 256));
    return offset += (20 + 256) * size_Uint8;
  };

  Model.prototype.getEnglishBoneNames = function(buffer, view, offset) {
    var block, i, length;
    length = this.bones.length;
    block = new Uint8Array(buffer, offset, 20 * length);
    this.english_bone_names = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(sjisArrayToString(slice.call(block, i * 20, (i + 1) * 20)));
      }
      return _results;
    })();
    return offset += length * 20 * size_Uint8;
  };

  Model.prototype.getEnglishMorphNames = function(buffer, view, offset) {
    var block, i, length;
    length = this.morphs.length - 1;
    if (length < 0) {
      length = 0;
    }
    block = new Uint8Array(buffer, offset, 20 * length);
    this.english_morph_names = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(sjisArrayToString(slice.call(block, i * 20, (i + 1) * 20)));
      }
      return _results;
    })();
    return offset += length * 20 * size_Uint8;
  };

  Model.prototype.getEnglishBoneGroupNames = function(buffer, view, offset) {
    var block, i, length;
    length = this.bone_group_names.length;
    block = new Uint8Array(buffer, offset, 50 * length);
    this.english_bone_group_names = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(sjisArrayToString(slice.call(block, i * 50, (i + 1) * 50)));
      }
      return _results;
    })();
    return offset += length * 50 * size_Uint8;
  };

  Model.prototype.getToonFileNames = function(buffer, view, offset) {
    var block, i;
    block = new Uint8Array(buffer, offset, 100 * 10);
    this.toon_file_names = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; _i < 10; i = ++_i) {
        _results.push(sjisArrayToString(slice.call(block, i * 100, (i + 1) * 100)));
      }
      return _results;
    })();
    return offset += 100 * 10 * size_Uint8;
  };

  Model.prototype.getRigidBodies = function(buffer, view, offset) {
    var i, length;
    length = view.getUint32(offset, true);
    offset += size_Uint32;
    this.rigid_bodies = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(new RigidBody(buffer, view, offset + i * RigidBody.size));
      }
      return _results;
    })();
    return offset += length * RigidBody.size;
  };

  Model.prototype.getJoints = function(buffer, view, offset) {
    var i, length;
    length = view.getUint32(offset, true);
    offset += size_Uint32;
    this.joints = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(new Joint(buffer, view, offset + i * Joint.size));
      }
      return _results;
    })();
    return offset += length * Joint.size;
  };

  return Model;

})();

Vertex = (function() {
  function Vertex(buffer, view, offset) {
    this.x = view.getFloat32(offset, true);
    offset += size_Float32;
    this.y = view.getFloat32(offset, true);
    offset += size_Float32;
    this.z = -view.getFloat32(offset, true);
    offset += size_Float32;
    this.nx = view.getFloat32(offset, true);
    offset += size_Float32;
    this.ny = view.getFloat32(offset, true);
    offset += size_Float32;
    this.nz = -view.getFloat32(offset, true);
    offset += size_Float32;
    this.u = view.getFloat32(offset, true);
    offset += size_Float32;
    this.v = view.getFloat32(offset, true);
    offset += size_Float32;
    this.bone_num1 = view.getUint16(offset, true);
    offset += size_Uint16;
    this.bone_num2 = view.getUint16(offset, true);
    offset += size_Uint16;
    this.bone_weight = view.getUint8(offset);
    offset += size_Uint8;
    this.edge_flag = view.getUint8(offset);
    offset += size_Uint8;
  }

  return Vertex;

})();

Vertex.size = size_Float32 * 8 + size_Uint16 * 2 + size_Uint8 * 2;

Material = (function() {
  function Material(buffer, view, offset) {
    var i, tmp;
    tmp = [];
    tmp[0] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = view.getFloat32(offset, true);
    offset += size_Float32;
    this.diffuse = new Float32Array(tmp);
    this.alpha = view.getFloat32(offset, true);
    offset += size_Float32;
    this.shininess = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[0] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = view.getFloat32(offset, true);
    offset += size_Float32;
    this.specular = new Float32Array(tmp);
    tmp[0] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = view.getFloat32(offset, true);
    offset += size_Float32;
    this.ambient = new Float32Array(tmp);
    this.toon_index = view.getInt8(offset);
    offset += size_Int8;
    this.edge_flag = view.getUint8(offset);
    offset += size_Uint8;
    this.face_vert_count = view.getUint32(offset, true);
    offset += size_Uint32;
    this.texture_file_name = sjisArrayToString((function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; _i < 20; i = ++_i) {
        _results.push(view.getUint8(offset + size_Uint8 * i));
      }
      return _results;
    })());
  }

  return Material;

})();

Material.size = size_Float32 * 11 + size_Uint8 * 2 + size_Uint32 + size_Uint8 * 20;

Bone = (function() {
  function Bone(buffer, view, offset) {
    var tmp;
    this.name = sjisArrayToString(new Uint8Array(buffer, offset, 20));
    offset += size_Uint8 * 20;
    this.parent_bone_index = view.getUint16(offset, true);
    offset += size_Uint16;
    this.tail_pos_bone_index = view.getUint16(offset, true);
    offset += size_Uint16;
    this.type = view.getUint8(offset);
    offset += size_Uint8;
    this.ik_parent_bone_index = view.getUint16(offset, true);
    offset += size_Uint16;
    tmp = [];
    tmp[0] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = -view.getFloat32(offset, true);
    offset += size_Float32;
    this.head_pos = new Float32Array(tmp);
  }

  return Bone;

})();

Bone.size = size_Uint8 * 21 + size_Uint16 * 3 + size_Float32 * 3;

IK = (function() {
  function IK(buffer, view, offset) {
    var chain_length, i;
    this.bone_index = view.getUint16(offset, true);
    offset += size_Uint16;
    this.target_bone_index = view.getUint16(offset, true);
    offset += size_Uint16;
    chain_length = view.getUint8(offset);
    offset += size_Uint8;
    this.iterations = view.getUint16(offset, true);
    offset += size_Uint16;
    this.control_weight = view.getFloat32(offset, true);
    offset += size_Float32;
    this.child_bones = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= chain_length ? _i < chain_length : _i > chain_length; i = 0 <= chain_length ? ++_i : --_i) {
        _results.push(view.getUint16(offset + size_Uint16 * i, true));
      }
      return _results;
    })();
  }

  IK.prototype.getSize = function() {
    return size_Uint16 * 3 + size_Uint8 + size_Float32 + size_Uint16 * this.child_bones.length;
  };

  return IK;

})();

Morph = (function() {
  function Morph(buffer, view, offset) {
    var data, i, vert_count;
    this.name = sjisArrayToString(new Uint8Array(buffer, offset, 20));
    offset += size_Uint8 * 20;
    vert_count = view.getUint32(offset, true);
    offset += size_Uint32;
    this.type = view.getUint8(offset);
    offset += size_Uint8;
    this.vert_data = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= vert_count ? _i < vert_count : _i > vert_count; i = 0 <= vert_count ? ++_i : --_i) {
        data = {};
        data.index = view.getUint32(offset, true);
        offset += size_Uint32;
        data.x = view.getFloat32(offset, true);
        offset += size_Float32;
        data.y = view.getFloat32(offset, true);
        offset += size_Float32;
        data.z = -view.getFloat32(offset, true);
        offset += size_Float32;
        _results.push(data);
      }
      return _results;
    })();
  }

  Morph.prototype.getSize = function() {
    return size_Uint8 * 21 + size_Uint32 + (size_Uint32 + size_Float32 * 3) * this.vert_data.length;
  };

  return Morph;

})();

RigidBody = (function() {
  function RigidBody(buffer, view, offset) {
    var tmp;
    this.name = sjisArrayToString(new Uint8Array(buffer, offset, 20));
    offset += size_Uint8 * 20;
    this.rel_bone_index = view.getUint16(offset, true);
    offset += size_Uint16;
    this.group_index = view.getUint8(offset);
    offset += size_Uint8;
    this.group_target = view.getUint8(offset);
    offset += size_Uint8;
    this.shape_type = view.getUint8(offset, true);
    offset += size_Uint8;
    this.shape_w = view.getFloat32(offset, true);
    offset += size_Float32;
    this.shape_h = view.getFloat32(offset, true);
    offset += size_Float32;
    this.shape_d = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp = [];
    tmp[0] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = -view.getFloat32(offset, true);
    offset += size_Float32;
    this.pos = new Float32Array(tmp);
    tmp[0] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = view.getFloat32(offset, true);
    offset += size_Float32;
    this.rot = new Float32Array(tmp);
    this.weight = view.getFloat32(offset, true);
    offset += size_Float32;
    this.pos_dim = view.getFloat32(offset, true);
    offset += size_Float32;
    this.rot_dim = view.getFloat32(offset, true);
    offset += size_Float32;
    this.recoil = view.getFloat32(offset, true);
    offset += size_Float32;
    this.friction = view.getFloat32(offset, true);
    offset += size_Float32;
    this.type = view.getUint8(offset);
    offset += size_Uint8;
  }

  return RigidBody;

})();

RigidBody.size = size_Uint8 * 23 + size_Uint16 * 2 + size_Float32 * 14;

Joint = (function() {
  function Joint(buffer, view, offset) {
    var tmp;
    this.name = sjisArrayToString(new Uint8Array(buffer, offset, 20));
    offset += size_Uint8 * 20;
    this.rigidbody_a = view.getUint32(offset, true);
    offset += size_Uint32;
    this.rigidbody_b = view.getUint32(offset, true);
    offset += size_Uint32;
    tmp = [];
    tmp[0] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = -view.getFloat32(offset, true);
    offset += size_Float32;
    this.pos = new Float32Array(tmp);
    tmp[0] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = view.getFloat32(offset, true);
    offset += size_Float32;
    this.rot = new Float32Array(tmp);
    tmp[0] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = -view.getFloat32(offset, true);
    offset += size_Float32;
    this.constrain_pos_1 = new Float32Array(tmp);
    tmp[0] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = -view.getFloat32(offset, true);
    offset += size_Float32;
    this.constrain_pos_2 = new Float32Array(tmp);
    tmp[0] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = view.getFloat32(offset, true);
    offset += size_Float32;
    this.constrain_rot_1 = new Float32Array(tmp);
    tmp[0] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = view.getFloat32(offset, true);
    offset += size_Float32;
    this.constrain_rot_2 = new Float32Array(tmp);
    tmp[0] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = -view.getFloat32(offset, true);
    offset += size_Float32;
    this.spring_pos = new Float32Array(tmp);
    tmp[0] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = view.getFloat32(offset, true);
    offset += size_Float32;
    this.spring_rot = new Float32Array(tmp);
  }

  return Joint;

})();

Joint.size = size_Int8 * 20 + size_Uint32 * 2 + size_Float32 * 24;

size_Uint8 = Uint8Array.BYTES_PER_ELEMENT;

size_Uint32 = Uint32Array.BYTES_PER_ELEMENT;

size_Float32 = Float32Array.BYTES_PER_ELEMENT;

slice = Array.prototype.slice;

this.MMD.Motion = (function() {
  function Motion(path) {
    this.path = path;
  }

  Motion.prototype.load = function(callback) {
    var xhr;
    xhr = new XMLHttpRequest;
    xhr.open('GET', this.path, true);
    xhr.responseType = 'arraybuffer';
    xhr.onload = (function(_this) {
      return function() {
        console.time('parse');
        _this.parse(xhr.response);
        console.timeEnd('parse');
        return callback();
      };
    })(this);
    return xhr.send();
  };

  Motion.prototype.parse = function(buffer) {
    var length, offset, view;
    length = buffer.byteLength;
    view = new DataView(buffer, 0);
    offset = 0;
    offset = this.checkHeader(buffer, view, offset);
    offset = this.getModelName(buffer, view, offset);
    offset = this.getBoneMotion(buffer, view, offset);
    offset = this.getMorphMotion(buffer, view, offset);
    offset = this.getCameraMotion(buffer, view, offset);
    offset = this.getLightMotion(buffer, view, offset);
    return offset = this.getSelfShadowMotion(buffer, view, offset);
  };

  Motion.prototype.checkHeader = function(buffer, view, offset) {
    if ('Vocaloid Motion Data 0002\0\0\0\0\0' !== String.fromCharCode.apply(null, slice.call(new Uint8Array(buffer, offset, 30)))) {
      throw 'File is not VMD';
    }
    return offset += 30 * size_Uint8;
  };

  Motion.prototype.getModelName = function(buffer, view, offset) {
    this.model_name = sjisArrayToString(new Uint8Array(buffer, offset, 20));
    return offset += size_Uint8 * 20;
  };

  Motion.prototype.getBoneMotion = function(buffer, view, offset) {
    var i, length;
    length = view.getUint32(offset, true);
    offset += size_Uint32;
    this.bone = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(new BoneMotion(buffer, view, offset + i * BoneMotion.size));
      }
      return _results;
    })();
    return offset += length * BoneMotion.size;
  };

  Motion.prototype.getMorphMotion = function(buffer, view, offset) {
    var i, length;
    length = view.getUint32(offset, true);
    offset += size_Uint32;
    this.morph = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(new MorphMotion(buffer, view, offset + i * MorphMotion.size));
      }
      return _results;
    })();
    return offset += length * MorphMotion.size;
  };

  Motion.prototype.getCameraMotion = function(buffer, view, offset) {
    var i, length;
    length = view.getUint32(offset, true);
    offset += size_Uint32;
    this.camera = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(new CameraMotion(buffer, view, offset + i * CameraMotion.size));
      }
      return _results;
    })();
    return offset += length * CameraMotion.size;
  };

  Motion.prototype.getLightMotion = function(buffer, view, offset) {
    var i, length;
    length = view.getUint32(offset, true);
    offset += size_Uint32;
    this.light = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(new LightMotion(buffer, view, offset + i * LightMotion.size));
      }
      return _results;
    })();
    return offset += length * LightMotion.size;
  };

  Motion.prototype.getSelfShadowMotion = function(buffer, view, offset) {
    var i, length;
    length = view.getUint32(offset, true);
    offset += size_Uint32;
    this.selfshadow = (function() {
      var _i, _results;
      _results = [];
      for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
        _results.push(new SelfShadowMotion(buffer, view, offset + i * SelfShadowMotion.size));
      }
      return _results;
    })();
    return offset += length * SelfShadowMotion.size;
  };

  return Motion;

})();

BoneMotion = (function() {
  function BoneMotion(buffer, view, offset) {
    var i, tmp, _i;
    this.name = sjisArrayToString(new Uint8Array(buffer, offset, 15));
    offset += size_Uint8 * 15;
    this.frame = view.getUint32(offset, true);
    offset += size_Uint32;
    tmp = [];
    tmp[0] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = -view.getFloat32(offset, true);
    offset += size_Float32;
    this.location = new Float32Array(tmp);
    tmp[0] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[3] = view.getFloat32(offset, true);
    offset += size_Float32;
    this.rotation = new Float32Array(tmp);
    for (i = _i = 0; _i < 64; i = ++_i) {
      tmp[i] = view.getUint8(offset, true);
      offset += size_Uint8;
    }
    this.interpolation = new Uint8Array(tmp);
  }

  return BoneMotion;

})();

BoneMotion.size = size_Uint8 * (15 + 64) + size_Uint32 + size_Float32 * 7;

MorphMotion = (function() {
  function MorphMotion(buffer, view, offset) {
    this.name = sjisArrayToString(new Uint8Array(buffer, offset, 15));
    offset += size_Uint8 * 15;
    this.frame = view.getUint32(offset, true);
    offset += size_Uint32;
    this.weight = view.getFloat32(offset, true);
    offset += size_Float32;
  }

  return MorphMotion;

})();

MorphMotion.size = size_Uint8 * 15 + size_Uint32 + size_Float32;

CameraMotion = (function() {
  function CameraMotion(buffer, view, offset) {
    var i, tmp, _i;
    this.frame = view.getUint32(offset, true);
    offset += size_Uint32;
    this.distance = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp = [];
    tmp[0] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = -view.getFloat32(offset, true);
    offset += size_Float32;
    this.location = new Float32Array(tmp);
    tmp[0] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = view.getFloat32(offset, true);
    offset += size_Float32;
    this.rotation = new Float32Array(tmp);
    for (i = _i = 0; _i < 24; i = ++_i) {
      tmp[i] = view.getUint8(offset, true);
      offset += size_Uint8;
    }
    this.interpolation = new Uint8Array(tmp);
    this.view_angle = view.getUint32(offset, true);
    offset += size_Uint32;
    this.noPerspective = view.getUint8(offset, true);
    offset += size_Uint8;
  }

  return CameraMotion;

})();

CameraMotion.size = size_Float32 * 7 + size_Uint8 * 25 + size_Float32 * 2;

LightMotion = (function() {
  function LightMotion(buffer, view, offset) {
    var tmp;
    this.frame = view.getUint32(offset, true);
    offset += size_Uint32;
    tmp = [];
    tmp[0] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = view.getFloat32(offset, true);
    offset += size_Float32;
    this.color = new Float32Array(tmp);
    tmp = [];
    tmp[0] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[1] = -view.getFloat32(offset, true);
    offset += size_Float32;
    tmp[2] = view.getFloat32(offset, true);
    offset += size_Float32;
    this.location = new Float32Array(tmp);
  }

  return LightMotion;

})();

LightMotion.size = size_Float32 * 6 + size_Uint32;

SelfShadowMotion = (function() {
  function SelfShadowMotion(buffer, view, offset) {
    this.frame = view.getUint32(offset, true);
    offset += size_Uint32;
    this.mode = view.getUint8(offset, true);
    offset += size_Uint8;
    this.distance = view.getFloat32(offset, true);
    offset += size_Float32;
  }

  return SelfShadowMotion;

})();

SelfShadowMotion.size = size_Float32 + size_Uint8 + size_Float32;

MMD.MotionManager = (function() {
  function MotionManager() {
    this.modelMotions = [];
    this.cameraMotion = [];
    this.cameraFrames = [];
    this.lightMotion = [];
    this.lightFrames = [];
    this.lastFrame = 0;
    return;
  }

  MotionManager.prototype.addModelMotion = function(model, motion, merge_flag, frame_offset) {
    var i, mm, _i, _len, _ref;
    _ref = this.modelMotions;
    for (i = _i = 0, _len = _ref.length; _i < _len; i = ++_i) {
      mm = _ref[i];
      if (model === mm.model) {
        break;
      }
    }
    if (i === this.modelMotions.length) {
      mm = new ModelMotion(model);
      this.modelMotions.push(mm);
    }
    mm.addBoneMotion(motion.bone, merge_flag, frame_offset);
    mm.addMorphMotion(motion.morph, merge_flag, frame_offset);
    this.lastFrame = mm.lastFrame;
  };

  MotionManager.prototype.getModelFrame = function(model, frame) {
    var i, mm, _i, _len, _ref;
    _ref = this.modelMotions;
    for (i = _i = 0, _len = _ref.length; _i < _len; i = ++_i) {
      mm = _ref[i];
      if (model === mm.model) {
        break;
      }
    }
    if (i === this.modelMotions.length) {
      return {};
    }
    return {
      bones: mm.getBoneFrame(frame),
      morphs: mm.getMorphFrame(frame)
    };
  };

  MotionManager.prototype.addCameraLightMotion = function(motion, merge_flag, frame_offset) {
    this.addCameraMotoin(motion.camera, merge_flag, frame_offset);
    this.addLightMotoin(motion.light, merge_flag, frame_offset);
  };

  MotionManager.prototype.addCameraMotoin = function(camera, merge_flag, frame_offset) {
    var c, frame, _i, _len;
    if (camera.length === 0) {
      return;
    }
    if (!merge_flag) {
      this.cameraMotion = [];
      this.cameraFrames = [];
    }
    frame_offset = frame_offset || 0;
    for (_i = 0, _len = camera.length; _i < _len; _i++) {
      c = camera[_i];
      frame = c.frame + frame_offset;
      this.cameraMotion[frame] = c;
      this.cameraFrames.push(frame);
      if (this.lastFrame < frame) {
        this.lastFrame = frame;
      }
    }
    this.cameraFrames = this.cameraFrames.sort(function(a, b) {
      return a - b;
    });
  };

  MotionManager.prototype.addLightMotoin = function(light, merge_flag, frame_offset) {
    var frame, l, _i, _len;
    if (light.length === 0) {
      return;
    }
    if (!merge_flag) {
      this.lightMotion = [];
      this.lightFrames = [];
    }
    frame_offset = frame_offset || 0;
    for (_i = 0, _len = light.length; _i < _len; _i++) {
      l = light[_i];
      frame = l.frame + frame_offset;
      this.lightMotion[frame] = l;
      this.lightFrames.push(frame);
      if (this.lastFrame < frame) {
        this.lastFrame = frame;
      }
    }
    this.lightFrames = this.lightFrames.sort(function(a, b) {
      return a - b;
    });
  };

  MotionManager.prototype.getCameraFrame = function(frame) {
    var bez, cache, camera, frac, frames, idx, lastFrame, n, next, p, prev, timeline;
    if (!this.cameraMotion.length) {
      return null;
    }
    timeline = this.cameraMotion;
    frames = this.cameraFrames;
    lastFrame = frames[frames.length - 1];
    if (lastFrame <= frame) {
      camera = timeline[lastFrame];
    } else {
      idx = previousRegisteredFrame(frames, frame);
      p = frames[idx];
      n = frames[idx + 1];
      frac = fraction(frame, p, n);
      prev = timeline[p];
      next = timeline[n];
      cache = [];
      bez = function(i) {
        var X1, X2, Y1, Y2, id;
        X1 = next.interpolation[i * 4];
        X2 = next.interpolation[i * 4 + 1];
        Y1 = next.interpolation[i * 4 + 2];
        Y2 = next.interpolation[i * 4 + 3];
        id = X1 | (X2 << 8) | (Y1 << 16) | (Y2 << 24);
        if (cache[id] != null) {
          return cache[id];
        }
        if (X1 === Y1 && X2 === Y2) {
          return cache[id] = frac;
        }
        return cache[id] = bezierp(X1 / 127, X2 / 127, Y1 / 127, Y2 / 127, frac);
      };
      camera = {
        location: vec3.createLerp3(prev.location, next.location, [bez(0), bez(1), bez(2)]),
        rotation: vec3.createLerp(prev.rotation, next.rotation, bez(3)),
        distance: lerp1(prev.distance, next.distance, bez(4)),
        view_angle: lerp1(prev.view_angle, next.view_angle, bez(5))
      };
    }
    return camera;
  };

  MotionManager.prototype.getLightFrame = function(frame) {
    var frac, frames, idx, lastFrame, light, n, next, p, prev, timeline;
    if (!this.lightMotion.length) {
      return null;
    }
    timeline = this.lightMotion;
    frames = this.lightFrames;
    lastFrame = frames[frames.length - 1];
    if (lastFrame <= frame) {
      light = timeline[lastFrame];
    } else {
      idx = previousRegisteredFrame(frames, frame);
      p = frames[idx];
      n = frames[idx + 1];
      frac = fraction(frame, p, n);
      prev = timeline[p];
      next = timeline[n];
      light = {
        color: vec3.createLerp(prev.color, next.color, frac),
        location: vec3.lerp(prev.location, next.location, frac)
      };
    }
    return light;
  };

  return MotionManager;

})();

ModelMotion = (function() {
  function ModelMotion(model) {
    this.model = model;
    this.boneMotions = {};
    this.boneFrames = {};
    this.morphMotions = {};
    this.morphFrames = {};
    this.lastFrame = 0;
  }

  ModelMotion.prototype.addBoneMotion = function(bone, merge_flag, frame_offset) {
    var b, frame, name, _i, _len;
    if (!merge_flag) {
      this.boneMotions = {};
      this.boneFrames = {};
    }
    frame_offset = frame_offset || 0;
    for (_i = 0, _len = bone.length; _i < _len; _i++) {
      b = bone[_i];
      if (!this.boneMotions[b.name]) {
        this.boneMotions[b.name] = [
          {
            location: vec3.create(),
            rotation: quat4.create([0, 0, 0, 1])
          }
        ];
      }
      frame = b.frame + frame_offset;
      this.boneMotions[b.name][frame] = b;
      if (this.lastFrame < frame) {
        this.lastFrame = frame;
      }
    }
    for (name in this.boneMotions) {
      this.boneFrames[name] = (this.boneFrames[name] || []).concat(Object.keys(this.boneMotions[name]).map(Number)).sort(function(a, b) {
        return a - b;
      });
    }
  };

  ModelMotion.prototype.addMorphMotion = function(morph, merge_flag, frame_offset) {
    var frame, m, name, _i, _len;
    if (!merge_flag) {
      this.morphMotions = {};
      this.morphFrames = {};
    }
    frame_offset = frame_offset || 0;
    for (_i = 0, _len = morph.length; _i < _len; _i++) {
      m = morph[_i];
      if (m.name === 'base') {
        continue;
      }
      if (!this.morphMotions[m.name]) {
        this.morphMotions[m.name] = [0];
      }
      frame = m.frame + frame_offset;
      this.morphMotions[m.name][frame] = m.weight;
      if (this.lastFrame < frame) {
        this.lastFrame = frame;
      }
    }
    for (name in this.morphMotions) {
      this.morphFrames[name] = (this.morphFrames[name] || []).concat(Object.keys(this.morphMotions[name]).map(Number)).sort(function(a, b) {
        return a - b;
      });
    }
  };

  ModelMotion.prototype.getBoneFrame = function(frame) {
    var bez, bones, cache, frac, frames, idx, lastFrame, n, name, next, p, prev, r, rotation, timeline;
    bones = {};
    for (name in this.boneMotions) {
      timeline = this.boneMotions[name];
      frames = this.boneFrames[name];
      lastFrame = frames[frames.length - 1];
      if (lastFrame <= frame) {
        bones[name] = timeline[lastFrame];
      } else {
        idx = previousRegisteredFrame(frames, frame);
        p = frames[idx];
        n = frames[idx + 1];
        frac = fraction(frame, p, n);
        prev = timeline[p];
        next = timeline[n];
        cache = [];
        bez = function(i) {
          var X1, X2, Y1, Y2, id;
          X1 = next.interpolation[i * 4];
          X2 = next.interpolation[i * 4 + 1];
          Y1 = next.interpolation[i * 4 + 2];
          Y2 = next.interpolation[i * 4 + 3];
          id = X1 | (X2 << 8) | (Y1 << 16) | (Y2 << 24);
          if (cache[id] != null) {
            return cache[id];
          }
          if (X1 === Y1 && X2 === Y2) {
            return cache[id] = frac;
          }
          return cache[id] = bezierp(X1 / 127, X2 / 127, Y1 / 127, Y2 / 127, frac);
        };
        if (quat4.dot(prev.rotation, next.rotation) >= 0) {
          rotation = quat4.createSlerp(prev.rotation, next.rotation, bez(3));
        } else {
          r = prev.rotation;
          rotation = quat4.createSlerp([-r[0], -r[1], -r[2], -r[3]], next.rotation, bez(3));
        }
        bones[name] = {
          location: vec3.createLerp3(prev.location, next.location, [bez(0), bez(1), bez(2)]),
          rotation: rotation
        };
      }
    }
    return bones;
  };

  ModelMotion.prototype.getMorphFrame = function(frame) {
    var frac, frames, idx, lastFrame, morphs, n, name, next, p, prev, timeline;
    morphs = {};
    for (name in this.morphMotions) {
      timeline = this.morphMotions[name];
      frames = this.morphFrames[name];
      lastFrame = frames[frames.length - 1];
      if (lastFrame <= frame) {
        morphs[name] = timeline[lastFrame];
      } else {
        idx = previousRegisteredFrame(frames, frame);
        p = frames[idx];
        n = frames[idx + 1];
        frac = fraction(frame, p, n);
        prev = timeline[p];
        next = timeline[n];
        morphs[name] = lerp1(prev, next, frac);
      }
    }
    return morphs;
  };

  return ModelMotion;

})();

previousRegisteredFrame = function(frames, frame) {

  /*
    'frames' is key frames registered, 'frame' is the key frame I'm enquiring about
    ex. frames: [0,10,20,30,40,50], frame: 15
    now I want to find the numbers 10 and 20, namely the ones before 15 and after 15
    I'm doing a bisection search here.
   */
  var delta, idx;
  idx = 0;
  delta = frames.length;
  while (true) {
    delta = (delta >> 1) || 1;
    if (frames[idx] <= frame) {
      if (delta === 1 && frames[idx + 1] > frame) {
        break;
      }
      idx += delta;
    } else {
      idx -= delta;
      if (delta === 1 && frames[idx] <= frame) {
        break;
      }
    }
  }
  return idx;
};

fraction = function(x, x0, x1) {
  return (x - x0) / (x1 - x0);
};

lerp1 = function(x0, x1, a) {
  return x0 + a * (x1 - x0);
};

bezierp = function(x1, x2, y1, y2, x) {

  /*
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
   */
  var t, tt, v;
  t = x;
  while (true) {
    v = ipfunc(t, x1, x2) - x;
    if (v * v < 0.0000001) {
      break;
    }
    tt = ipfuncd(t, x1, x2);
    if (tt === 0) {
      break;
    }
    t -= v / tt;
  }
  return ipfunc(t, y1, y2);
};

ipfunc = function(t, p1, p2) {
  return (1 + 3 * p1 - 3 * p2) * t * t * t + (3 * p2 - 6 * p1) * t * t + 3 * p1 * t;
};

ipfuncd = function(t, p1, p2) {
  return (3 + 9 * p1 - 9 * p2) * t * t + (6 * p2 - 12 * p1) * t + 3 * p1;
};

MMD.ShadowMap = (function() {
  function ShadowMap(mmd) {
    this.mmd = mmd;
    this.framebuffer = this.texture = null;
    this.width = this.height = 2048;
    this.viewBroadness = 0.6;
    this.debug = false;
    this.initFramebuffer();
  }

  ShadowMap.prototype.initFramebuffer = function() {
    var gl, renderbuffer;
    gl = this.mmd.gl;
    this.framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    renderbuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, renderbuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, this.width, this.height);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, renderbuffer);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    return gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  };

  ShadowMap.prototype.computeMatrices = function() {
    var cameraPosition, center, cx, cy, lengthScale, lightDirection, size, viewMatrix;
    center = vec3.create(this.mmd.center);
    lightDirection = vec3.createNormalize(this.mmd.lightDirection);
    vec3.add(lightDirection, center);
    cameraPosition = vec3.create(this.mmd.cameraPosition);
    lengthScale = vec3.lengthBetween(cameraPosition, center);
    size = lengthScale * this.viewBroadness;
    viewMatrix = mat4.lookAt(lightDirection, center, [0, 1, 0]);
    this.mvMatrix = mat4.createMultiply(viewMatrix, this.mmd.modelMatrix);
    mat4.multiplyVec3(viewMatrix, center);
    cx = center[0];
    cy = center[1];
    this.pMatrix = mat4.ortho(cx - size, cx + size, cy - size, cy + size, -size, size);
  };

  ShadowMap.prototype.beforeRender = function() {
    var gl, program;
    gl = this.mmd.gl;
    program = this.mmd.program;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.viewport(0, 0, this.width, this.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.uniform1i(program.uGenerateShadowMap, true);
    gl.uniformMatrix4fv(program.uMVMatrix, false, this.mvMatrix);
    gl.uniformMatrix4fv(program.uPMatrix, false, this.pMatrix);
  };

  ShadowMap.prototype.afterRender = function() {
    var gl, program;
    gl = this.mmd.gl;
    program = this.mmd.program;
    gl.uniform1i(program.uGenerateShadowMap, false);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
    if (this.debug) {
      this.debugTexture();
    }
  };

  ShadowMap.prototype.getLightMatrix = function() {
    var lightMatrix;
    lightMatrix = mat4.createMultiply(this.pMatrix, this.mvMatrix);
    mat4.applyScale(lightMatrix, [0.5, 0.5, 0.5]);
    mat4.applyTranslate(lightMatrix, [0.5, 0.5, 0.5]);
    return lightMatrix;
  };

  ShadowMap.prototype.debugTexture = function() {
    var canvas, ctx, data, gl, i, imageData, pixelarray, _i, _ref;
    gl = this.mmd.gl;
    pixelarray = new Uint8Array(this.width * this.height * 4);
    gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.UNSIGNED_BYTE, pixelarray);
    canvas = document.getElementById('shadowmap');
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = 'shadowmap';
      canvas.width = this.width;
      canvas.height = this.height;
      canvas.style.border = 'solid black 1px';
      canvas.style.width = this.mmd.width + 'px';
      canvas.style.height = this.mmd.height + 'px';
      document.body.appendChild(canvas);
    }
    ctx = canvas.getContext('2d');
    imageData = ctx.getImageData(0, 0, this.width, this.height);
    data = imageData.data;
    for (i = _i = 0, _ref = data.length; 0 <= _ref ? _i < _ref : _i > _ref; i = 0 <= _ref ? ++_i : --_i) {
      data[i] = pixelarray[i];
    }
    return ctx.putImageData(imageData, 0, 0);
  };

  ShadowMap.prototype.getTexture = function() {
    return this.texture;
  };

  return ShadowMap;

})();

MMD.TextureManager = (function() {
  function TextureManager(mmd) {
    this.mmd = mmd;
    this.store = {};
    this.pendingCount = 0;
  }

  TextureManager.prototype.get = function(type, url) {
    var gl, texture;
    texture = this.store[url];
    if (texture) {
      return texture;
    }
    gl = this.mmd.gl;
    texture = this.store[url] = gl.createTexture();
    loadImage(url, (function(_this) {
      return function(img) {
        img = checkSize(img);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        if (type === 'toon') {
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        } else {
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
        }
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.bindTexture(gl.TEXTURE_2D, null);
        if (_this.onload) {
          _this.onload(img);
        }
        return --_this.pendingCount;
      };
    })(this));
    this.pendingCount++;
    return texture;
  };

  return TextureManager;

})();

checkSize = function(img) {
  var canv, h, size, w;
  w = img.naturalWidth;
  h = img.naturalHeight;
  size = 1 << (Math.log(Math.min(w, h)) / Math.LN2 | 0);
  if (w !== h || w !== size) {
    canv = document.createElement('canvas');
    canv.height = canv.width = size;
    canv.getContext('2d').drawImage(img, 0, 0, w, h, 0, 0, size, size);
    img = canv;
  }
  return img;
};

loadImage = function(url, callback) {
  var img;
  img = new Image;
  img.onload = function() {
    return callback(img);
  };
  img.onerror = function() {
    return alert('failed to load image: ' + url);
  };
  img.src = url;
  return img;
};

MMD.VertexShaderSource = '\nuniform mat4 uMVMatrix; // model-view matrix (model -> view space)\nuniform mat4 uPMatrix; // projection matrix (view -> projection space)\nuniform mat4 uNMatrix; // normal matrix (inverse of transpose of model-view matrix)\n\nuniform mat4 uLightMatrix; // mvpdMatrix of light space (model -> display space)\n\nattribute vec3 aVertexNormal;\nattribute vec2 aTextureCoord;\nattribute float aVertexEdge; // 0 or 1. 1 if the vertex has an edge. (becuase we can\'t pass bool to attributes)\n\nattribute float aBoneWeight;\nattribute vec3 aVectorFromBone1;\nattribute vec3 aVectorFromBone2;\nattribute vec4 aBone1Rotation;\nattribute vec4 aBone2Rotation;\nattribute vec3 aBone1Position;\nattribute vec3 aBone2Position;\n\nattribute vec3 aMultiPurposeVector;\n\nvarying vec3 vPosition;\nvarying vec3 vNormal;\nvarying vec2 vTextureCoord;\nvarying vec4 vLightCoord; // coordinate in light space; to be mapped onto shadow map\n\nuniform float uEdgeThickness;\nuniform bool uEdge;\n\nuniform bool uGenerateShadowMap;\n\nuniform bool uSelfShadow;\n\nuniform bool uAxis;\nuniform bool uCenterPoint;\n\nvec3 qtransform(vec4 q, vec3 v) {\n  return v + 2.0 * cross(cross(v, q.xyz) - q.w*v, q.xyz);\n}\n\nvoid main() {\n  vec3 position;\n  vec3 normal;\n\n  if (uAxis || uCenterPoint) {\n\n    position = aMultiPurposeVector;\n\n  } else {\n\n    float weight = aBoneWeight;\n    vec3 morph = aMultiPurposeVector;\n\n    position = qtransform(aBone1Rotation, aVectorFromBone1 + morph) + aBone1Position;\n    normal = qtransform(aBone1Rotation, aVertexNormal);\n\n    if (weight < 0.99) {\n      vec3 p2 = qtransform(aBone2Rotation, aVectorFromBone2 + morph) + aBone2Position;\n      vec3 n2 = qtransform(aBone2Rotation, normal);\n\n      position = mix(p2, position, weight);\n      normal = normalize(mix(n2, normal, weight));\n    }\n  }\n\n  // return vertex point in projection space\n  gl_Position = uPMatrix * uMVMatrix * vec4(position, 1.0);\n\n  if (uCenterPoint) {\n    gl_Position.z = 0.0; // always on top\n    gl_PointSize = 16.0;\n  }\n\n  if (uGenerateShadowMap || uAxis || uCenterPoint) return;\n\n  // for fragment shader\n  vTextureCoord = aTextureCoord;\n  vPosition = (uMVMatrix * vec4(position, 1.0)).xyz;\n  vNormal = (uNMatrix * vec4(normal, 1.0)).xyz;\n\n  if (uSelfShadow) {\n    vLightCoord = uLightMatrix * vec4(position, 1.0);\n  }\n\n  if (uEdge) {\n    vec4 pos = gl_Position;\n    vec4 pos2 = uPMatrix * uMVMatrix * vec4(position + normal, 1.0);\n    vec4 norm = normalize(pos2 - pos);\n    gl_Position = pos + norm * uEdgeThickness * aVertexEdge * pos.w; // scale by pos.w to prevent becoming thicker when zoomed\n    return;\n  }\n}\n';

this.MMD = (function() {
  function MMD(canvas, width, height) {
    this.width = width;
    this.height = height;
    this.gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!this.gl) {
      alert('WebGL not supported in your browser');
      throw 'WebGL not supported';
    }
  }

  MMD.prototype.initShaders = function() {
    var attributes, fshader, line, name, src, type, uniforms, vshader, _i, _j, _k, _l, _len, _len1, _len2, _len3, _ref, _ref1, _ref2;
    vshader = this.gl.createShader(this.gl.VERTEX_SHADER);
    this.gl.shaderSource(vshader, MMD.VertexShaderSource);
    this.gl.compileShader(vshader);
    if (!this.gl.getShaderParameter(vshader, this.gl.COMPILE_STATUS)) {
      alert('Vertex shader compilation error');
      throw this.gl.getShaderInfoLog(vshader);
    }
    fshader = this.gl.createShader(this.gl.FRAGMENT_SHADER);
    this.gl.shaderSource(fshader, MMD.FragmentShaderSource);
    this.gl.compileShader(fshader);
    if (!this.gl.getShaderParameter(fshader, this.gl.COMPILE_STATUS)) {
      alert('Fragment shader compilation error');
      throw this.gl.getShaderInfoLog(fshader);
    }
    this.program = this.gl.createProgram();
    this.gl.attachShader(this.program, vshader);
    this.gl.attachShader(this.program, fshader);
    this.gl.linkProgram(this.program);
    if (!this.gl.getProgramParameter(this.program, this.gl.LINK_STATUS)) {
      alert('Shader linking error');
      throw this.gl.getProgramInfoLog(this.program);
    }
    this.gl.useProgram(this.program);
    attributes = [];
    uniforms = [];
    _ref = [MMD.VertexShaderSource, MMD.FragmentShaderSource];
    for (_i = 0, _len = _ref.length; _i < _len; _i++) {
      src = _ref[_i];
      _ref1 = src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '').split(';');
      for (_j = 0, _len1 = _ref1.length; _j < _len1; _j++) {
        line = _ref1[_j];
        type = (_ref2 = line.match(/^\s*(uniform|attribute)\s+/)) != null ? _ref2[1] : void 0;
        if (!type) {
          continue;
        }
        name = line.match(/(\w+)(\[\d+\])?\s*$/)[1];
        if (type === 'attribute' && __indexOf.call(attributes, name) < 0) {
          attributes.push(name);
        }
        if (type === 'uniform' && __indexOf.call(uniforms, name) < 0) {
          uniforms.push(name);
        }
      }
    }
    for (_k = 0, _len2 = attributes.length; _k < _len2; _k++) {
      name = attributes[_k];
      this.program[name] = this.gl.getAttribLocation(this.program, name);
      this.gl.enableVertexAttribArray(this.program[name]);
    }
    for (_l = 0, _len3 = uniforms.length; _l < _len3; _l++) {
      name = uniforms[_l];
      this.program[name] = this.gl.getUniformLocation(this.program, name);
    }
  };

  MMD.prototype.addModel = function(model) {
    this.model = model;
  };

  MMD.prototype.initBuffers = function() {
    this.vbuffers = {};
    this.initVertices();
    this.initIndices();
    this.initTextures();
  };

  MMD.prototype.initVertices = function() {
    var bone1, bone2, buffer, data, edge, i, length, model, morphVec, normals, positions1, positions2, rotations1, rotations2, uvs, vectors1, vectors2, vertex, weight, _i, _j, _len, _ref;
    model = this.model;
    length = model.vertices.length;
    weight = new Float32Array(length);
    vectors1 = new Float32Array(3 * length);
    vectors2 = new Float32Array(3 * length);
    rotations1 = new Float32Array(4 * length);
    rotations2 = new Float32Array(4 * length);
    positions1 = new Float32Array(3 * length);
    positions2 = new Float32Array(3 * length);
    morphVec = new Float32Array(3 * length);
    normals = new Float32Array(3 * length);
    uvs = new Float32Array(2 * length);
    edge = new Float32Array(length);
    for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
      vertex = model.vertices[i];
      bone1 = model.bones[vertex.bone_num1];
      bone2 = model.bones[vertex.bone_num2];
      weight[i] = vertex.bone_weight;
      vectors1[3 * i] = vertex.x - bone1.head_pos[0];
      vectors1[3 * i + 1] = vertex.y - bone1.head_pos[1];
      vectors1[3 * i + 2] = vertex.z - bone1.head_pos[2];
      vectors2[3 * i] = vertex.x - bone2.head_pos[0];
      vectors2[3 * i + 1] = vertex.y - bone2.head_pos[1];
      vectors2[3 * i + 2] = vertex.z - bone2.head_pos[2];
      positions1[3 * i] = bone1.head_pos[0];
      positions1[3 * i + 1] = bone1.head_pos[1];
      positions1[3 * i + 2] = bone1.head_pos[2];
      positions2[3 * i] = bone2.head_pos[0];
      positions2[3 * i + 1] = bone2.head_pos[1];
      positions2[3 * i + 2] = bone2.head_pos[2];
      rotations1[4 * i + 3] = 1;
      rotations2[4 * i + 3] = 1;
      normals[3 * i] = vertex.nx;
      normals[3 * i + 1] = vertex.ny;
      normals[3 * i + 2] = vertex.nz;
      uvs[2 * i] = vertex.u;
      uvs[2 * i + 1] = vertex.v;
      edge[i] = 1 - vertex.edge_flag;
    }
    model.rotations1 = rotations1;
    model.rotations2 = rotations2;
    model.positions1 = positions1;
    model.positions2 = positions2;
    model.morphVec = morphVec;
    _ref = [
      {
        attribute: 'aBoneWeight',
        array: weight,
        size: 1
      }, {
        attribute: 'aVectorFromBone1',
        array: vectors1,
        size: 3
      }, {
        attribute: 'aVectorFromBone2',
        array: vectors2,
        size: 3
      }, {
        attribute: 'aBone1Rotation',
        array: rotations1,
        size: 4
      }, {
        attribute: 'aBone2Rotation',
        array: rotations2,
        size: 4
      }, {
        attribute: 'aBone1Position',
        array: positions1,
        size: 3
      }, {
        attribute: 'aBone2Position',
        array: positions2,
        size: 3
      }, {
        attribute: 'aMultiPurposeVector',
        array: morphVec,
        size: 3
      }, {
        attribute: 'aVertexNormal',
        array: normals,
        size: 3
      }, {
        attribute: 'aTextureCoord',
        array: uvs,
        size: 2
      }, {
        attribute: 'aVertexEdge',
        array: edge,
        size: 1
      }
    ];
    for (_j = 0, _len = _ref.length; _j < _len; _j++) {
      data = _ref[_j];
      buffer = this.gl.createBuffer();
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, data.array, this.gl.STATIC_DRAW);
      this.vbuffers[data.attribute] = {
        size: data.size,
        buffer: buffer
      };
    }
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
  };

  MMD.prototype.initIndices = function() {
    var indices;
    indices = this.model.triangles;
    this.ibuffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.ibuffer);
    this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, indices, this.gl.STATIC_DRAW);
    this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, null);
  };

  MMD.prototype.initTextures = function() {
    var fileName, material, model, toonIndex, type, _i, _j, _len, _len1, _ref, _ref1;
    model = this.model;
    this.textureManager = new MMD.TextureManager(this);
    this.textureManager.onload = (function(_this) {
      return function() {
        return _this.redraw = true;
      };
    })(this);
    _ref = model.materials;
    for (_i = 0, _len = _ref.length; _i < _len; _i++) {
      material = _ref[_i];
      if (!material.textures) {
        material.textures = {};
      }
      toonIndex = material.toon_index;
      fileName = 'toon' + ('0' + (toonIndex + 1)).slice(-2) + '.bmp';
      if (toonIndex === -1 || !model.toon_file_names || fileName === model.toon_file_names[toonIndex]) {
        fileName = 'lib/MMD.js/data/' + fileName;
      } else {
        fileName = model.directory + '/' + model.toon_file_names[toonIndex];
      }
      material.textures.toon = this.textureManager.get('toon', fileName);
      if (material.texture_file_name) {
        _ref1 = material.texture_file_name.split('*');
        for (_j = 0, _len1 = _ref1.length; _j < _len1; _j++) {
          fileName = _ref1[_j];
          switch (fileName.slice(-4)) {
            case '.sph':
              type = 'sph';
              break;
            case '.spa':
              type = 'spa';
              break;
            case '.tga':
              type = 'regular';
              fileName += '.png';
              break;
            default:
              type = 'regular';
          }
          material.textures[type] = this.textureManager.get(type, model.directory + '/' + fileName);
        }
      }
    }
  };

  MMD.prototype.start = function() {
    var before, count, interval, step, t0;
    this.gl.clearColor(1, 1, 1, 1);
    this.gl.clearDepth(1);
    this.gl.enable(this.gl.DEPTH_TEST);
    this.redraw = true;
    if (this.drawSelfShadow) {
      this.shadowMap = new MMD.ShadowMap(this);
    }
    this.motionManager = new MMD.MotionManager;
    count = 0;
    t0 = before = Date.now();
    interval = 1000 / this.fps;
    step = (function(_this) {
      return function() {
        var now;
        _this.move();
        _this.computeMatrices();
        _this.render();
        now = Date.now();
        if (++count % _this.fps === 0) {
          _this.realFps = _this.fps / (now - before) * 1000;
          before = now;
        }
        return setTimeout(step, (t0 + count * interval) - now);
      };
    })(this);
    step();
  };

  MMD.prototype.move = function() {
    if (!this.playing || this.textureManager.pendingCount > 0) {
      return;
    }
    if (++this.frame > this.motionManager.lastFrame) {
      this.pause();
      this.frame = -1;
      return;
    }
    this.moveCamera();
    this.moveLight();
    this.moveModel();
  };

  MMD.prototype.moveCamera = function() {
    var camera;
    camera = this.motionManager.getCameraFrame(this.frame);
    if (camera && !this.ignoreCameraMotion) {
      this.distance = camera.distance;
      this.rotx = camera.rotation[0];
      this.roty = camera.rotation[1];
      this.center = vec3.create(camera.location);
      this.fovy = camera.view_angle;
    }
  };

  MMD.prototype.moveLight = function() {
    var light;
    light = this.motionManager.getLightFrame(this.frame);
    if (light) {
      this.lightDirection = light.location;
      this.lightColor = light.color;
    }
  };

  MMD.prototype.moveModel = function() {
    var bones, model, morphs, _ref;
    model = this.model;
    _ref = this.motionManager.getModelFrame(model, this.frame), morphs = _ref.morphs, bones = _ref.bones;
    this.moveMorphs(model, morphs);
    this.moveBones(model, bones);
  };

  MMD.prototype.moveMorphs = function(model, morphs) {
    var b, base, i, j, morph, vert, weight, _i, _j, _k, _len, _len1, _len2, _ref, _ref1, _ref2;
    if (!morphs) {
      return;
    }
    if (model.morphs.length === 0) {
      return;
    }
    _ref = model.morphs;
    for (j = _i = 0, _len = _ref.length; _i < _len; j = ++_i) {
      morph = _ref[j];
      if (j === 0) {
        base = morph;
        continue;
      }
      if (!(morph.name in morphs)) {
        continue;
      }
      weight = morphs[morph.name];
      _ref1 = morph.vert_data;
      for (_j = 0, _len1 = _ref1.length; _j < _len1; _j++) {
        vert = _ref1[_j];
        b = base.vert_data[vert.index];
        i = b.index;
        model.morphVec[3 * i] += vert.x * weight;
        model.morphVec[3 * i + 1] += vert.y * weight;
        model.morphVec[3 * i + 2] += vert.z * weight;
      }
    }
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vbuffers.aMultiPurposeVector.buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, model.morphVec, this.gl.STATIC_DRAW);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
    _ref2 = base.vert_data;
    for (_k = 0, _len2 = _ref2.length; _k < _len2; _k++) {
      b = _ref2[_k];
      i = b.index;
      model.morphVec[3 * i] = 0;
      model.morphVec[3 * i + 1] = 0;
      model.morphVec[3 * i + 2] = 0;
    }
  };

  MMD.prototype.moveBones = function(model, bones) {
    var bone, boneMotions, constrainedBones, getBoneMotion, i, individualBoneMotions, length, motion1, motion2, originalBonePositions, parentBones, pos1, pos2, positions1, positions2, resolveIKs, rot1, rot2, rotations1, rotations2, vertex, _i, _j, _k, _len, _ref, _ref1, _ref2;
    if (!bones) {
      return;
    }
    individualBoneMotions = [];
    boneMotions = [];
    originalBonePositions = [];
    parentBones = [];
    constrainedBones = [];
    _ref = model.bones;
    for (i = _i = 0, _len = _ref.length; _i < _len; i = ++_i) {
      bone = _ref[i];
      individualBoneMotions[i] = (_ref1 = bones[bone.name]) != null ? _ref1 : {
        rotation: quat4.create([0, 0, 0, 1]),
        location: vec3.create()
      };
      boneMotions[i] = {
        r: quat4.create(),
        p: vec3.create(),
        tainted: true
      };
      originalBonePositions[i] = bone.head_pos;
      parentBones[i] = bone.parent_bone_index;
      if (bone.name.indexOf('\u3072\u3056') > 0) {
        constrainedBones[i] = true;
      }
    }
    getBoneMotion = function(boneIndex) {
      var m, motion, p, parentIndex, parentMotion, r, t;
      motion = boneMotions[boneIndex];
      if (motion && !motion.tainted) {
        return motion;
      }
      m = individualBoneMotions[boneIndex];
      r = quat4.set(m.rotation, motion.r);
      t = m.location;
      p = vec3.set(originalBonePositions[boneIndex], motion.p);
      if (parentBones[boneIndex] === 0xFFFF) {
        return boneMotions[boneIndex] = {
          p: vec3.add(p, t),
          r: r,
          tainted: false
        };
      } else {
        parentIndex = parentBones[boneIndex];
        parentMotion = getBoneMotion(parentIndex);
        r = quat4.multiply(parentMotion.r, r, r);
        p = vec3.subtract(p, originalBonePositions[parentIndex]);
        vec3.add(p, t);
        vec3.rotateByQuat4(p, parentMotion.r);
        vec3.add(p, parentMotion.p);
        return boneMotions[boneIndex] = {
          p: p,
          r: r,
          tainted: false
        };
      }
    };
    resolveIKs = function() {
      var axis, axisLen, boneIndex, bonePos, c, ik, ikbonePos, ikboneVec, ikboneVecLen, j, maxangle, minLength, motion, n, parentRotation, q, r, sinTheta, targetIndex, targetPos, targetVec, targetVecLen, theta, tmpQ, tmpR, _j, _len1, _ref2, _results;
      targetVec = vec3.create();
      ikboneVec = vec3.create();
      axis = vec3.create();
      tmpQ = quat4.create();
      tmpR = quat4.create();
      _ref2 = model.iks;
      _results = [];
      for (_j = 0, _len1 = _ref2.length; _j < _len1; _j++) {
        ik = _ref2[_j];
        ikbonePos = getBoneMotion(ik.bone_index).p;
        targetIndex = ik.target_bone_index;
        minLength = 0.1 * vec3.length(vec3.subtract(originalBonePositions[targetIndex], originalBonePositions[parentBones[targetIndex]], axis));
        _results.push((function() {
          var _k, _ref3, _results1;
          _results1 = [];
          for (n = _k = 0, _ref3 = ik.iterations; 0 <= _ref3 ? _k < _ref3 : _k > _ref3; n = 0 <= _ref3 ? ++_k : --_k) {
            targetPos = getBoneMotion(targetIndex).p;
            if (minLength > vec3.length(vec3.subtract(targetPos, ikbonePos, axis))) {
              break;
            }
            _results1.push((function() {
              var _l, _len2, _m, _ref4, _results2;
              _ref4 = ik.child_bones;
              _results2 = [];
              for (i = _l = 0, _len2 = _ref4.length; _l < _len2; i = ++_l) {
                boneIndex = _ref4[i];
                motion = getBoneMotion(boneIndex);
                bonePos = motion.p;
                if (i > 0) {
                  targetPos = getBoneMotion(targetIndex).p;
                }
                targetVec = vec3.subtract(targetPos, bonePos, targetVec);
                targetVecLen = vec3.length(targetVec);
                if (targetVecLen < minLength) {
                  continue;
                }
                ikboneVec = vec3.subtract(ikbonePos, bonePos, ikboneVec);
                ikboneVecLen = vec3.length(ikboneVec);
                if (ikboneVecLen < minLength) {
                  continue;
                }
                axis = vec3.cross(targetVec, ikboneVec, axis);
                axisLen = vec3.length(axis);
                sinTheta = axisLen / ikboneVecLen / targetVecLen;
                if (sinTheta < 0.001) {
                  continue;
                }
                maxangle = (i + 1) * ik.control_weight * 4;
                theta = Math.asin(sinTheta);
                if (vec3.dot(targetVec, ikboneVec) < 0) {
                  theta = 3.141592653589793 - theta;
                }
                if (theta > maxangle) {
                  theta = maxangle;
                }
                q = quat4.set(vec3.scale(axis, Math.sin(theta / 2) / axisLen), tmpQ);
                q[3] = Math.cos(theta / 2);
                parentRotation = getBoneMotion(parentBones[boneIndex]).r;
                r = quat4.inverse(parentRotation, tmpR);
                r = quat4.multiply(quat4.multiply(r, q), motion.r);
                if (constrainedBones[boneIndex]) {
                  c = r[3];
                  r = quat4.set([Math.sqrt(1 - c * c), 0, 0, c], r);
                  quat4.inverse(boneMotions[boneIndex].r, q);
                  quat4.multiply(r, q, q);
                  q = quat4.multiply(parentRotation, q, q);
                }
                quat4.normalize(r, individualBoneMotions[boneIndex].rotation);
                quat4.multiply(q, motion.r, motion.r);
                for (j = _m = 0; 0 <= i ? _m < i : _m > i; j = 0 <= i ? ++_m : --_m) {
                  boneMotions[ik.child_bones[j]].tainted = true;
                }
                _results2.push(boneMotions[ik.target_bone_index].tainted = true);
              }
              return _results2;
            })());
          }
          return _results1;
        })());
      }
      return _results;
    };
    resolveIKs();
    for (i = _j = 0, _ref2 = model.bones.length; 0 <= _ref2 ? _j < _ref2 : _j > _ref2; i = 0 <= _ref2 ? ++_j : --_j) {
      getBoneMotion(i);
    }
    rotations1 = model.rotations1;
    rotations2 = model.rotations2;
    positions1 = model.positions1;
    positions2 = model.positions2;
    length = model.vertices.length;
    for (i = _k = 0; 0 <= length ? _k < length : _k > length; i = 0 <= length ? ++_k : --_k) {
      vertex = model.vertices[i];
      motion1 = boneMotions[vertex.bone_num1];
      motion2 = boneMotions[vertex.bone_num2];
      rot1 = motion1.r;
      pos1 = motion1.p;
      rot2 = motion2.r;
      pos2 = motion2.p;
      rotations1[i * 4] = rot1[0];
      rotations1[i * 4 + 1] = rot1[1];
      rotations1[i * 4 + 2] = rot1[2];
      rotations1[i * 4 + 3] = rot1[3];
      rotations2[i * 4] = rot2[0];
      rotations2[i * 4 + 1] = rot2[1];
      rotations2[i * 4 + 2] = rot2[2];
      rotations2[i * 4 + 3] = rot2[3];
      positions1[i * 3] = pos1[0];
      positions1[i * 3 + 1] = pos1[1];
      positions1[i * 3 + 2] = pos1[2];
      positions2[i * 3] = pos2[0];
      positions2[i * 3 + 1] = pos2[1];
      positions2[i * 3 + 2] = pos2[2];
    }
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vbuffers.aBone1Rotation.buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, rotations1, this.gl.STATIC_DRAW);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vbuffers.aBone2Rotation.buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, rotations2, this.gl.STATIC_DRAW);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vbuffers.aBone1Position.buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, positions1, this.gl.STATIC_DRAW);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vbuffers.aBone2Position.buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, positions2, this.gl.STATIC_DRAW);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
  };

  MMD.prototype.computeMatrices = function() {
    var up;
    this.modelMatrix = mat4.createIdentity();
    this.cameraPosition = vec3.create([0, 0, this.distance]);
    vec3.rotateX(this.cameraPosition, this.rotx);
    vec3.rotateY(this.cameraPosition, this.roty);
    vec3.moveBy(this.cameraPosition, this.center);
    up = [0, 1, 0];
    vec3.rotateX(up, this.rotx);
    vec3.rotateY(up, this.roty);
    this.viewMatrix = mat4.lookAt(this.cameraPosition, this.center, up);
    this.mvMatrix = mat4.createMultiply(this.viewMatrix, this.modelMatrix);
    this.pMatrix = mat4.perspective(this.fovy, this.width / this.height, 0.1, 1000.0);
    this.nMatrix = mat4.inverseTranspose(this.mvMatrix, mat4.create());
  };

  MMD.prototype.render = function() {
    var attribute, material, offset, vb, _i, _j, _len, _len1, _ref, _ref1, _ref2;
    if (!this.redraw && !this.playing) {
      return;
    }
    this.redraw = false;
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    this.gl.viewport(0, 0, this.width, this.height);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
    _ref = this.vbuffers;
    for (attribute in _ref) {
      vb = _ref[attribute];
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vb.buffer);
      this.gl.vertexAttribPointer(this.program[attribute], vb.size, this.gl.FLOAT, false, 0, 0);
    }
    this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.ibuffer);
    this.setSelfShadowTexture();
    this.setUniforms();
    this.gl.enable(this.gl.CULL_FACE);
    this.gl.enable(this.gl.BLEND);
    this.gl.blendFuncSeparate(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA, this.gl.SRC_ALPHA, this.gl.DST_ALPHA);
    offset = 0;
    _ref1 = this.model.materials;
    for (_i = 0, _len = _ref1.length; _i < _len; _i++) {
      material = _ref1[_i];
      this.renderMaterial(material, offset);
      offset += material.face_vert_count;
    }
    this.gl.disable(this.gl.BLEND);
    offset = 0;
    _ref2 = this.model.materials;
    for (_j = 0, _len1 = _ref2.length; _j < _len1; _j++) {
      material = _ref2[_j];
      this.renderEdge(material, offset);
      offset += material.face_vert_count;
    }
    this.gl.disable(this.gl.CULL_FACE);
    this.renderAxes();
    this.gl.flush();
  };

  MMD.prototype.setSelfShadowTexture = function() {
    var material, model, offset, _i, _len, _ref, _ref1;
    if (!this.drawSelfShadow) {
      return;
    }
    model = this.model;
    this.shadowMap.computeMatrices();
    this.shadowMap.beforeRender();
    offset = 0;
    _ref = model.materials;
    for (_i = 0, _len = _ref.length; _i < _len; _i++) {
      material = _ref[_i];
      if ((0.979 < (_ref1 = material.alpha) && _ref1 < 0.981)) {
        continue;
      }
      this.gl.drawElements(this.gl.TRIANGLES, material.face_vert_count, this.gl.UNSIGNED_SHORT, offset * 2);
      offset += material.face_vert_count;
    }
    this.shadowMap.afterRender();
    this.gl.activeTexture(this.gl.TEXTURE3);
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.shadowMap.getTexture());
    this.gl.uniform1i(this.program.uShadowMap, 3);
    this.gl.uniformMatrix4fv(this.program.uLightMatrix, false, this.shadowMap.getLightMatrix());
    this.gl.uniform1i(this.program.uSelfShadow, true);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    this.gl.viewport(0, 0, this.width, this.height);
  };

  MMD.prototype.setUniforms = function() {
    var lightDirection;
    this.gl.uniform1f(this.program.uEdgeThickness, this.edgeThickness);
    this.gl.uniform3fv(this.program.uEdgeColor, this.edgeColor);
    this.gl.uniformMatrix4fv(this.program.uMVMatrix, false, this.mvMatrix);
    this.gl.uniformMatrix4fv(this.program.uPMatrix, false, this.pMatrix);
    this.gl.uniformMatrix4fv(this.program.uNMatrix, false, this.nMatrix);
    lightDirection = vec3.createNormalize(this.lightDirection);
    mat4.multiplyVec3(this.nMatrix, lightDirection);
    this.gl.uniform3fv(this.program.uLightDirection, lightDirection);
    this.gl.uniform3fv(this.program.uLightColor, this.lightColor);
  };

  MMD.prototype.renderMaterial = function(material, offset) {
    var textures;
    this.gl.uniform3fv(this.program.uAmbientColor, material.ambient);
    this.gl.uniform3fv(this.program.uSpecularColor, material.specular);
    this.gl.uniform3fv(this.program.uDiffuseColor, material.diffuse);
    this.gl.uniform1f(this.program.uAlpha, material.alpha);
    this.gl.uniform1f(this.program.uShininess, material.shininess);
    this.gl.uniform1i(this.program.uEdge, false);
    textures = material.textures;
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, textures.toon);
    this.gl.uniform1i(this.program.uToon, 0);
    if (textures.regular) {
      this.gl.activeTexture(this.gl.TEXTURE1);
      this.gl.bindTexture(this.gl.TEXTURE_2D, textures.regular);
      this.gl.uniform1i(this.program.uTexture, 1);
    }
    this.gl.uniform1i(this.program.uUseTexture, !!textures.regular);
    if (textures.sph || textures.spa) {
      this.gl.activeTexture(this.gl.TEXTURE2);
      this.gl.bindTexture(this.gl.TEXTURE_2D, textures.sph || textures.spa);
      this.gl.uniform1i(this.program.uSphereMap, 2);
      this.gl.uniform1i(this.program.uUseSphereMap, true);
      this.gl.uniform1i(this.program.uIsSphereMapAdditive, !!textures.spa);
    } else {
      this.gl.uniform1i(this.program.uUseSphereMap, false);
    }
    this.gl.cullFace(this.gl.BACK);
    this.gl.drawElements(this.gl.TRIANGLES, material.face_vert_count, this.gl.UNSIGNED_SHORT, offset * 2);
  };

  MMD.prototype.renderEdge = function(material, offset) {
    if (!this.drawEdge || !material.edge_flag) {
      return;
    }
    this.gl.uniform1i(this.program.uEdge, true);
    this.gl.cullFace(this.gl.FRONT);
    this.gl.drawElements(this.gl.TRIANGLES, material.face_vert_count, this.gl.UNSIGNED_SHORT, offset * 2);
    this.gl.cullFace(this.gl.BACK);
    return this.gl.uniform1i(this.program.uEdge, false);
  };

  MMD.prototype.renderAxes = function() {
    var axis, axisBuffer, color, i, _i, _j;
    axisBuffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, axisBuffer);
    this.gl.vertexAttribPointer(this.program.aMultiPurposeVector, 3, this.gl.FLOAT, false, 0, 0);
    if (this.drawAxes) {
      this.gl.uniform1i(this.program.uAxis, true);
      for (i = _i = 0; _i < 3; i = ++_i) {
        axis = [0, 0, 0, 0, 0, 0];
        axis[i] = 65;
        color = [0, 0, 0];
        color[i] = 1;
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(axis), this.gl.STATIC_DRAW);
        this.gl.uniform3fv(this.program.uAxisColor, color);
        this.gl.drawArrays(this.gl.LINES, 0, 2);
      }
      axis = [-50, 0, 0, 0, 0, 0, 0, 0, -50, 0, 0, 0];
      for (i = _j = -50; _j <= 50; i = _j += 5) {
        if (i !== 0) {
          axis.push(i, 0, -50, i, 0, 50, -50, 0, i, 50, 0, i);
        }
      }
      color = [0.7, 0.7, 0.7];
      this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(axis), this.gl.STATIC_DRAW);
      this.gl.uniform3fv(this.program.uAxisColor, color);
      this.gl.drawArrays(this.gl.LINES, 0, 84);
      this.gl.uniform1i(this.program.uAxis, false);
    }
    if (this.drawCenterPoint) {
      this.gl.uniform1i(this.program.uCenterPoint, true);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(this.center), this.gl.STATIC_DRAW);
      this.gl.drawArrays(this.gl.POINTS, 0, 1);
      this.gl.uniform1i(this.program.uCenterPoint, false);
    }
    this.gl.deleteBuffer(axisBuffer);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
  };

  MMD.prototype.registerKeyListener = function(element) {
    element.addEventListener('keydown', (function(_this) {
      return function(e) {
        switch (e.keyCode + e.shiftKey * 1000 + e.ctrlKey * 10000 + e.altKey * 100000) {
          case 37:
            _this.roty += Math.PI / 12;
            break;
          case 39:
            _this.roty -= Math.PI / 12;
            break;
          case 38:
            _this.rotx += Math.PI / 12;
            break;
          case 40:
            _this.rotx -= Math.PI / 12;
            break;
          case 33:
            _this.distance -= 3 * _this.distance / _this.DIST;
            break;
          case 34:
            _this.distance += 3 * _this.distance / _this.DIST;
            break;
          case 36:
            _this.rotx = _this.roty = 0;
            _this.center = [0, 10, 0];
            _this.distance = _this.DIST;
            break;
          case 1037:
            vec3.multiplyMat4(_this.center, _this.mvMatrix);
            _this.center[0] -= _this.distance / _this.DIST;
            vec3.multiplyMat4(_this.center, mat4.createInverse(_this.mvMatrix));
            break;
          case 1039:
            vec3.multiplyMat4(_this.center, _this.mvMatrix);
            _this.center[0] += _this.distance / _this.DIST;
            vec3.multiplyMat4(_this.center, mat4.createInverse(_this.mvMatrix));
            break;
          case 1038:
            vec3.multiplyMat4(_this.center, _this.mvMatrix);
            _this.center[1] += _this.distance / _this.DIST;
            vec3.multiplyMat4(_this.center, mat4.createInverse(_this.mvMatrix));
            break;
          case 1040:
            vec3.multiplyMat4(_this.center, _this.mvMatrix);
            _this.center[1] -= _this.distance / _this.DIST;
            vec3.multiplyMat4(_this.center, mat4.createInverse(_this.mvMatrix));
            break;
          case 32:
            if (_this.playing) {
              _this.pause();
            } else {
              _this.play();
            }
            break;
          default:
            return;
        }
        e.preventDefault();
        return _this.redraw = true;
      };
    })(this), false);
  };

  MMD.prototype.registerMouseListener = function(element) {
    this.registerDragListener(element);
    this.registerWheelListener(element);
  };

  MMD.prototype.registerDragListener = function(element) {
    element.addEventListener('mousedown', (function(_this) {
      return function(e) {
        var modifier, move, onmousemove, onmouseup, ox, oy;
        if (e.button !== 0) {
          return;
        }
        modifier = e.shiftKey * 1000 + e.ctrlKey * 10000 + e.altKey * 100000;
        if (modifier !== 0 && modifier !== 1000) {
          return;
        }
        ox = e.clientX;
        oy = e.clientY;
        move = function(dx, dy, modi) {
          if (modi === 0) {
            _this.roty -= dx / 100;
            _this.rotx -= dy / 100;
            return _this.redraw = true;
          } else if (modi === 1000) {
            vec3.multiplyMat4(_this.center, _this.mvMatrix);
            _this.center[0] -= dx / 30 * _this.distance / _this.DIST;
            _this.center[1] += dy / 30 * _this.distance / _this.DIST;
            vec3.multiplyMat4(_this.center, mat4.createInverse(_this.mvMatrix));
            return _this.redraw = true;
          }
        };
        onmouseup = function(e) {
          var modi;
          if (e.button !== 0) {
            return;
          }
          modi = e.shiftKey * 1000 + e.ctrlKey * 10000 + e.altKey * 100000;
          move(e.clientX - ox, e.clientY - oy, modi);
          element.removeEventListener('mouseup', onmouseup, false);
          element.removeEventListener('mousemove', onmousemove, false);
          return e.preventDefault();
        };
        onmousemove = function(e) {
          var modi, x, y;
          if (e.button !== 0) {
            return;
          }
          modi = e.shiftKey * 1000 + e.ctrlKey * 10000 + e.altKey * 100000;
          x = e.clientX;
          y = e.clientY;
          move(x - ox, y - oy, modi);
          ox = x;
          oy = y;
          return e.preventDefault();
        };
        element.addEventListener('mouseup', onmouseup, false);
        return element.addEventListener('mousemove', onmousemove, false);
      };
    })(this), false);
  };

  MMD.prototype.registerWheelListener = function(element) {
    var onwheel;
    onwheel = (function(_this) {
      return function(e) {
        var delta;
        delta = e.detail || e.wheelDelta / (-40);
        _this.distance += delta * _this.distance / _this.DIST;
        _this.redraw = true;
        return e.preventDefault();
      };
    })(this);
    if ('onmousewheel' in window) {
      element.addEventListener('mousewheel', onwheel, false);
    } else {
      element.addEventListener('DOMMouseScroll', onwheel, false);
    }
  };

  MMD.prototype.initParameters = function() {
    this.ignoreCameraMotion = false;
    this.rotx = this.roty = 0;
    this.distance = this.DIST = 35;
    this.center = [0, 10, 0];
    this.fovy = 40;
    this.drawEdge = true;
    this.edgeThickness = 0.004;
    this.edgeColor = [0, 0, 0];
    this.lightDirection = [0.5, 1.0, 0.5];
    this.lightDistance = 8875;
    this.lightColor = [0.6, 0.6, 0.6];
    this.drawSelfShadow = true;
    this.drawAxes = true;
    this.drawCenterPoint = false;
    this.fps = 30;
    this.realFps = this.fps;
    this.playing = false;
    this.frame = -1;
  };

  MMD.prototype.addCameraLightMotion = function(motion, merge_flag, frame_offset) {
    this.motionManager.addCameraLightMotion(motion, merge_flag, frame_offset);
  };

  MMD.prototype.addModelMotion = function(model, motion, merge_flag, frame_offset) {
    this.motionManager.addModelMotion(model, motion, merge_flag, frame_offset);
  };

  MMD.prototype.play = function() {
    this.playing = true;
  };

  MMD.prototype.pause = function() {
    this.playing = false;
  };

  MMD.prototype.rewind = function() {
    this.setFrameNumber(-1);
  };

  MMD.prototype.setFrameNumber = function(num) {
    this.frame = num;
  };

  return MMD;

})();
