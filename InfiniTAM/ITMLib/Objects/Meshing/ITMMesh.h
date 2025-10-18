// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Scene/ITMVoxelBlockHash.h"
#include "../../../ORUtils/Image.h"

#include <stdlib.h>
#include <vector>
#include <map>

namespace ITMLib
{
	class ITMMesh
	{
	public:
		// struct Triangle { Vector3f p0, p1, p2; };
		struct Triangle
		{
			Vector3f p0, p1, p2, c0, c1, c2, clr;
		};
		MemoryDeviceType memoryType;

		uint noTotalTriangles;
		static const uint noMaxTriangles_default = SDF_LOCAL_BLOCK_NUM * 32 * 16;
		uint noMaxTriangles;

		ORUtils::MemoryBlock<Triangle> *triangles;

		explicit ITMMesh(MemoryDeviceType memoryType, uint maxTriangles = noMaxTriangles_default)
		{
			this->memoryType = memoryType;
			this->noTotalTriangles = 0;
			this->noMaxTriangles = maxTriangles;

			triangles = new ORUtils::MemoryBlock<Triangle>(noMaxTriangles, memoryType);
		}

		void WritePLY(const char *fileName)
		{
			printf("write ply mesh...\n");
			ORUtils::MemoryBlock<Triangle> *cpu_triangles;
			bool shouldDelete = false;
			if (memoryType == MEMORYDEVICE_CUDA)
			{
				cpu_triangles = new ORUtils::MemoryBlock<Triangle>(noMaxTriangles, MEMORYDEVICE_CPU);
				cpu_triangles->SetFrom(triangles, ORUtils::MemoryBlock<Triangle>::CUDA_TO_CPU);
				shouldDelete = true;
			}
			else
				cpu_triangles = triangles;

			Triangle *triangleArray = cpu_triangles->GetData(MEMORYDEVICE_CPU);

			FILE *f = fopen(fileName, "w");

			if (f != NULL)
			{
				// 写入PLY文件头
				fprintf(f, "ply\n");
				fprintf(f, "format ascii 1.0\n");
				fprintf(f, "element vertex %d\n", noTotalTriangles * 3);
				fprintf(f, "property float x\n");
				fprintf(f, "property float y\n");
				fprintf(f, "property float z\n");
				fprintf(f, "property uchar red\n");
				fprintf(f, "property uchar green\n");
				fprintf(f, "property uchar blue\n");
				fprintf(f, "element face %d\n", noTotalTriangles);
				fprintf(f, "property list uchar int vertex_indices\n");
				fprintf(f, "end_header\n");

				// 写入顶点数据
				for (uint i = 0; i < noTotalTriangles; i++)
				{
					fprintf(f, "%f %f %f %d %d %d\n",
							triangleArray[i].p0.x, triangleArray[i].p0.y, triangleArray[i].p0.z,
							static_cast<unsigned char>(triangleArray[i].c0.r * 255),
							static_cast<unsigned char>(triangleArray[i].c0.g * 255),
							static_cast<unsigned char>(triangleArray[i].c0.b * 255));
					fprintf(f, "%f %f %f %d %d %d\n",
							triangleArray[i].p1.x, triangleArray[i].p1.y, triangleArray[i].p1.z,
							static_cast<unsigned char>(triangleArray[i].c1.r * 255),
							static_cast<unsigned char>(triangleArray[i].c1.g * 255),
							static_cast<unsigned char>(triangleArray[i].c1.b * 255));
					fprintf(f, "%f %f %f %d %d %d\n",
							triangleArray[i].p2.x, triangleArray[i].p2.y, triangleArray[i].p2.z,
							static_cast<unsigned char>(triangleArray[i].c2.r * 255),
							static_cast<unsigned char>(triangleArray[i].c2.g * 255),
							static_cast<unsigned char>(triangleArray[i].c2.b * 255));
				}

				// 写入面数据
				for (uint i = 0; i < noTotalTriangles; i++)
				{
					fprintf(f, "3 %d %d %d\n", i * 3, i * 3 + 1, i * 3 + 2);
				}

				fclose(f);
			}

			if (shouldDelete)
			{
				delete cpu_triangles;
			}
		}

		// NOTE: 这个代码有bug，保存出来的obj文件无法使用meshlab可视化
		void WriteOBJ(const char *fileName)
		{
			ORUtils::MemoryBlock<Triangle> *cpu_triangles;
			bool shoulDelete = false;
			if (memoryType == MEMORYDEVICE_CUDA)
			{
				cpu_triangles = new ORUtils::MemoryBlock<Triangle>(noMaxTriangles, MEMORYDEVICE_CPU);
				cpu_triangles->SetFrom(triangles, ORUtils::MemoryBlock<Triangle>::CUDA_TO_CPU);
				shoulDelete = true;
			}
			else
				cpu_triangles = triangles;

			Triangle *triangleArray = cpu_triangles->GetData(MEMORYDEVICE_CPU);

			FILE *f = fopen(fileName, "w+");
			if (f != NULL)
			{
				for (uint i = 0; i < noTotalTriangles; i++)
				{
					// fprintf(f, "v %f %f %f\n", triangleArray[i].p0.x, triangleArray[i].p0.y, triangleArray[i].p0.z);
					// fprintf(f, "v %f %f %f\n", triangleArray[i].p1.x, triangleArray[i].p1.y, triangleArray[i].p1.z);
					// fprintf(f, "v %f %f %f\n", triangleArray[i].p2.x, triangleArray[i].p2.y, triangleArray[i].p2.z);
					// const Vector3f clr = triangleArray[i].clr;
					fprintf(f, "v %f %f %f %f %f %f\n",
							triangleArray[i].p0.x, triangleArray[i].p0.y, triangleArray[i].p0.z,
							triangleArray[i].c0.r, triangleArray[i].c0.g, triangleArray[i].c0.b);
					fprintf(f, "v %f %f %f %f %f %f\n",
							triangleArray[i].p1.x, triangleArray[i].p1.y, triangleArray[i].p1.z,
							triangleArray[i].c1.r, triangleArray[i].c1.g, triangleArray[i].c1.b);
					fprintf(f, "v %f %f %f %f %f %f\n",
							triangleArray[i].p2.x, triangleArray[i].p2.y, triangleArray[i].p2.z,
							triangleArray[i].c2.r, triangleArray[i].c2.g, triangleArray[i].c2.b);
				}

				for (uint i = 0; i < noTotalTriangles; i++)
					fprintf(f, "f %d %d %d\n", i * 3 + 2 + 1, i * 3 + 1 + 1, i * 3 + 0 + 1);
				fclose(f);
			}

			if (shoulDelete)
				delete cpu_triangles;
		}

		void WriteSTL(const char *fileName)
		{
			ORUtils::MemoryBlock<Triangle> *cpu_triangles;
			bool shoulDelete = false;
			if (memoryType == MEMORYDEVICE_CUDA)
			{
				cpu_triangles = new ORUtils::MemoryBlock<Triangle>(noMaxTriangles, MEMORYDEVICE_CPU);
				cpu_triangles->SetFrom(triangles, ORUtils::MemoryBlock<Triangle>::CUDA_TO_CPU);
				shoulDelete = true;
			}
			else
				cpu_triangles = triangles;

			Triangle *triangleArray = cpu_triangles->GetData(MEMORYDEVICE_CPU);

			FILE *f = fopen(fileName, "wb+");

			if (f != NULL)
			{
				for (int i = 0; i < 80; i++)
					fwrite(" ", sizeof(char), 1, f);

				fwrite(&noTotalTriangles, sizeof(int), 1, f);

				float zero = 0.0f;
				short attribute = 0;
				for (uint i = 0; i < noTotalTriangles; i++)
				{
					fwrite(&zero, sizeof(float), 1, f);
					fwrite(&zero, sizeof(float), 1, f);
					fwrite(&zero, sizeof(float), 1, f);

					fwrite(&triangleArray[i].p2.x, sizeof(float), 1, f);
					fwrite(&triangleArray[i].p2.y, sizeof(float), 1, f);
					fwrite(&triangleArray[i].p2.z, sizeof(float), 1, f);

					fwrite(&triangleArray[i].p1.x, sizeof(float), 1, f);
					fwrite(&triangleArray[i].p1.y, sizeof(float), 1, f);
					fwrite(&triangleArray[i].p1.z, sizeof(float), 1, f);

					fwrite(&triangleArray[i].p0.x, sizeof(float), 1, f);
					fwrite(&triangleArray[i].p0.y, sizeof(float), 1, f);
					fwrite(&triangleArray[i].p0.z, sizeof(float), 1, f);

					fwrite(&attribute, sizeof(short), 1, f);

					// fprintf(f, "v %f %f %f\n", triangleArray[i].p0.x, triangleArray[i].p0.y, triangleArray[i].p0.z);
					// fprintf(f, "v %f %f %f\n", triangleArray[i].p1.x, triangleArray[i].p1.y, triangleArray[i].p1.z);
					// fprintf(f, "v %f %f %f\n", triangleArray[i].p2.x, triangleArray[i].p2.y, triangleArray[i].p2.z);
				}

				// for (uint i = 0; i<noTotalTriangles; i++) fprintf(f, "f %d %d %d\n", i * 3 + 2 + 1, i * 3 + 1 + 1, i * 3 + 0 + 1);
				fclose(f);
			}

			if (shoulDelete)
				delete cpu_triangles;
		}

		~ITMMesh()
		{
			delete triangles;
		}

		// Suppress the default copy constructor and assignment operator
		ITMMesh(const ITMMesh &);
		ITMMesh &operator=(const ITMMesh &);
	};
}
