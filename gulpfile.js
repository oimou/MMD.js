var gulp = require('gulp')
var coffee = require('gulp-coffee')
var concat = require('gulp-concat')

gulp.task('build', function () {
  gulp.src('src/*.coffee')
    .pipe(concat('mmd-concat.coffee'))
    .pipe(gulp.dest('.'))
    .pipe(coffee({bare: true}))
    .pipe(gulp.dest('MMD.js'))
})

gulp.task('watch', function () {
  gulp.src('src/*.coffee', ['build'])
})

gulp.task('default', ['build'])
