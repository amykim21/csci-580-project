#include	"gz.h"
#ifndef GZRENDER_
#define GZRENDER_


/* Camera defaults */
#define	DEFAULT_FOV		35.0
#define	DEFAULT_IM_Z	(-10.0)  /* world coords for image plane origin */
#define	DEFAULT_IM_Y	(5.0)    /* default look-at point = 0,0,0 */
#define	DEFAULT_IM_X	(-10.0)

#define	DEFAULT_AMBIENT	{0.1, 0.1, 0.1}
#define	DEFAULT_DIFFUSE	{0.7, 0.6, 0.5}
#define	DEFAULT_SPECULAR	{0.2, 0.3, 0.4}
#define	DEFAULT_SPEC		32

#define	MATLEVELS	100		/* how many matrix pushes allowed */
#define	MAX_LIGHTS	10		/* how many lights allowed */
#define	MAX_TRIANGLES	10000		/* how many triangles allowed */
#define	MAX_SPHERES	10

class GzRender{			/* define a renderer */
  

public:
	unsigned short	xres;
	unsigned short	yres;
	GzPixel		*pixelbuffer;		/* frame buffer array */
	char* framebuffer;
	short	triangleNum;
	short	sphereNum;
	GzTriangle triangles[MAX_TRIANGLES];
	GzCamera		m_camera;
	short		    matlevel;	        /* top of stack - current xform */
	GzMatrix		Ximage[MATLEVELS];	/* stack of xforms (Xsm) */
	GzMatrix		Xnorm[MATLEVELS];	/* xforms for norms (Xim) */
	GzMatrix		Xsp;		        /* NDC to screen (pers-to-screen) */
	GzColor		flatcolor;          /* color state for flat shaded triangles */
	int			interp_mode;
	int			numlights;
	int			numTriangles;
	GzLight		lights[MAX_LIGHTS];


	GzSphere lightSources[3];	// TODO: Make up param, need to move to rend.h later
	GzSphere	spheres[MAX_SPHERES];


	GzLight		ambientlight;
	GzColor		Ka, Kd, Ks;
	float		    spec;		/* specular power */
	GzTexture		tex_fun;    /* tex_fun(float u, float v, GzColor color) */
	
  	// Constructors
	GzRender(int xRes, int yRes);
	~GzRender();

	// Function declaration

	// HW1: Display methods
	int GzDefault();
	int GzBeginRender();
	int GzPut(int i, int j, GzIntensity r, GzIntensity g, GzIntensity b, GzIntensity a, GzDepth z);
	int GzGet(int i, int j, GzIntensity *r, GzIntensity *g, GzIntensity *b, GzIntensity *a, GzDepth	*z);

	int GzFlushDisplay2File(FILE* outfile);
	int GzFlushDisplay2FrameBuffer();

	// HW2: Render methods
	int GzPutAttribute(int numAttributes, GzToken *nameList, GzPointer *valueList);
	int GzPutTriangle(int numParts, GzToken* nameList, GzPointer* valueList);

	// HW3
	int GzPutCamera(GzCamera camera);
	int GzPushMatrix(GzMatrix	matrix);
	int GzPopMatrix();
	
	//bool isInShadow(GzVertex intersection, GzLight light);
	// Extra methods: NOT part of API - just for general assistance */
	inline int ARRAY(int x, int y){return (x+y*xres);}	/* simplify fbuf indexing */
	inline short	ctoi(float color) {return(short)((int)(color * ((1 << 12) - 1)));}		/* convert float color to GzIntensity short */


	// Object Translation
	int GzRotXMat(float degree, GzMatrix mat);
	int GzRotYMat(float degree, GzMatrix mat);
	int GzRotZMat(float degree, GzMatrix mat);
	int GzTrxMat(GzCoord translate, GzMatrix mat);
	int GzScaleMat(GzCoord scale, GzMatrix mat);

	void RayTrace();

	GzVector3D FresnelReflection(GzRay light, GzVertex intersection, GzTriangle triangle, int depth);
	GzVector3D EmitLight(GzRay ray, int depth);
	bool GzCollisionWithTriangle(GzRay light, int& index);
	bool GzCollisionWithSpecificTriangle(GzRay light, GzTriangle triangle, GzVector3D intersectPos);

	bool RayIntersectsSphere(const double origin[3], const double direction[3], const double center[3], double radius, double& t);
	bool GzCollisionWithSphere(GzRay light, int& index);
	void CalculateSphereReflectionAndRefraction(GzRay light, GzSphere sphere, const double intersectionPoint[3], GzRay& reflectLight, GzRay& refractLight);
};
#endif