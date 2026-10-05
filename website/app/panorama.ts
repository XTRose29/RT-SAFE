/** Small WebGL panorama renderer. Viewpoints are native Unreal cubemap captures. */
export type ViewAngle = { yaw: number; pitch: number; fov: number };
export class Panorama {
  private gl: WebGLRenderingContext;
  private program: WebGLProgram;
  private texture: WebGLTexture;
  private buffer: WebGLBuffer;
  private generation = 0;
  private disposed = false;
  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl", { alpha: false, antialias: false });
    if (!gl) throw new Error("360° viewing is unavailable in this browser.");
    this.gl = gl;
    const shader = (kind: number, source: string) => {
      const s = gl.createShader(kind)!; gl.shaderSource(s, source); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { const message = gl.getShaderInfoLog(s); gl.deleteShader(s); throw new Error(message || "Shader failed"); }
      return s;
    };
    const vert = shader(gl.VERTEX_SHADER, "attribute vec2 p; varying vec2 uv; void main(){uv=p;gl_Position=vec4(p,0.,1.);}");
    const frag = shader(gl.FRAGMENT_SHADER, `precision highp float;
      varying vec2 uv; uniform samplerCube scene; uniform vec3 angle; uniform float aspect;
      void main(){
        float y=angle.x, p=angle.y;
        vec3 forward=vec3(sin(y)*cos(p),sin(p),cos(y)*cos(p));
        vec3 right=vec3(cos(y),0.,-sin(y));
        vec3 up=vec3(-sin(y)*sin(p),cos(p),-cos(y)*sin(p));
        vec3 ray=forward+tan(angle.z*.5)*(right*uv.x*aspect+up*uv.y);
        gl_FragColor=textureCube(scene,normalize(ray));
      }`);
    this.program = gl.createProgram()!; gl.attachShader(this.program, vert); gl.attachShader(this.program, frag); gl.linkProgram(this.program);
    gl.deleteShader(vert); gl.deleteShader(frag);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error("Panorama program failed");
    gl.useProgram(this.program);
    this.buffer = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), gl.STATIC_DRAW);
    const p = gl.getAttribLocation(this.program, "p"); gl.enableVertexAttribArray(p); gl.vertexAttribPointer(p,2,gl.FLOAT,false,0,0);
    this.texture = gl.createTexture()!; gl.bindTexture(gl.TEXTURE_CUBE_MAP,this.texture);
    for(let i=0;i<6;i++) gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,0,gl.RGB,1,1,0,gl.RGB,gl.UNSIGNED_BYTE,new Uint8Array([30,46,43]));
    gl.texParameteri(gl.TEXTURE_CUBE_MAP,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  }
  async load(id: string) {
    const generation = ++this.generation;
    // Texture axes are (UE Y, UE Z, UE X). Cube faces retain native camera orientation.
    const images = await Promise.all(["py","ny","pz","nz","px","nx"].map(async face => {
      const im = new Image(); im.src=`media/nyc/explorer/${id}-${face}.webp`; await im.decode(); return im;
    }));
    if (this.disposed || generation !== this.generation) return false;
    const gl=this.gl; gl.bindTexture(gl.TEXTURE_CUBE_MAP,this.texture);
    images.forEach((im,i)=>gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,im));
    return true;
  }
  draw(angle: ViewAngle) {
    if (this.disposed) return;
    const gl=this.gl, dpr=Math.min(devicePixelRatio||1,2);
    const width=Math.max(1,Math.round(this.canvas.clientWidth*dpr)),height=Math.max(1,Math.round(this.canvas.clientHeight*dpr));
    if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
    gl.viewport(0,0,width,height);gl.useProgram(this.program);
    gl.uniform3f(gl.getUniformLocation(this.program,"angle"),angle.yaw*Math.PI/180,angle.pitch*Math.PI/180,angle.fov*Math.PI/180);
    gl.uniform1f(gl.getUniformLocation(this.program,"aspect"),width/height);gl.drawArrays(gl.TRIANGLES,0,6);
  }
  dispose() { this.disposed=true;this.generation++;this.gl.deleteTexture(this.texture);this.gl.deleteBuffer(this.buffer);this.gl.deleteProgram(this.program); }
}
